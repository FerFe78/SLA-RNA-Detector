# 7_analizar_impacto_k.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Importar la configuración unificada
import o_config as config

# --- Funciones Auxiliares ---
def crear_histograma(kmeans_model, descriptores_imagen, vocab_size):
    """Crea un histograma BoVW para una imagen."""
    if descriptores_imagen is None or len(descriptores_imagen) == 0:
        return np.zeros(vocab_size)
    palabras_visuales = kmeans_model.predict(descriptores_imagen)
    hist, _ = np.histogram(palabras_visuales, bins=np.arange(vocab_size + 1))
    hist = hist.astype(float)
    if np.sum(hist) > 0:
        hist /= np.linalg.norm(hist)
    return hist

def run_experiment_for_k(k_value, all_descriptors, labels, train_indices, test_indices):
    """Ejecuta el pipeline para un valor de k y devuelve el accuracy."""
    print(f"\n----- Ejecutando experimento para k = {k_value} -----")
    
    training_descriptors = [all_descriptors[i] for i in train_indices if all_descriptors[i] is not None]
    if not training_descriptors:
        print("  - No hay descriptores válidos en el conjunto de entrenamiento para este k. Saltando.")
        return 0.0 # Retornar un valor bajo
        
    kmeans = MiniBatchKMeans(n_clusters=k_value, random_state=config.RANDOM_STATE, n_init='auto').fit(np.vstack(training_descriptors))
    
    X_hist = np.array([crear_histograma(kmeans, desc, k_value) for desc in tqdm(all_descriptors, desc=f"Creando hist (k={k_value})")])
    y_hist = np.array([1 if label == 1 else 0 for label in labels])

    X_train, X_test = X_hist[train_indices], X_hist[test_indices]
    y_train, y_test = y_hist[train_indices], y_hist[test_indices]

    model = SVC(C=config.SVM_C, kernel=config.SVM_KERNEL, gamma=config.SVM_GAMMA, random_state=config.RANDOM_STATE).fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    print(f"  - Accuracy para k={k_value}: {accuracy:.4f}")
    return accuracy

def main():
    print("--- PASO 7: Análisis del Impacto de K y Entrenamiento del Mejor Modelo ---")
    
    desc_path = config.DESCRIPTORS_DIR / config.K_ANALYSIS_DESCRIPTORS_FILE
    if not desc_path.exists():
        print(f"ERROR: No se encontró '{desc_path.name}'.")
        return
        
    print(f"Cargando descriptores desde: {desc_path}")
    with open(desc_path, 'rb') as f:
        data = pickle.load(f)
    all_descriptors, labels = data['features'], data['labels']

    indices = np.arange(len(all_descriptors))
    train_indices, test_indices, _, _ = train_test_split(
        indices, labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=labels
    )

    accuracies = [run_experiment_for_k(k, all_descriptors, labels, train_indices, test_indices) for k in config.K_VALUES_TO_TEST]
        
    # --- Generación del Gráfico ---
    print("\n--- Generando gráfico de rendimiento vs. k ---")
    plt.figure(figsize=(12, 7))
    plt.plot(config.K_VALUES_TO_TEST, accuracies, 'o-', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=2)
    plt.title(f'Impacto del Tamaño del Vocabulario (k) en el Accuracy\n(Descriptor: {config.K_ANALYSIS_DESCRIPTORS_FILE})', fontsize=16)
    plt.xlabel('Tamaño del Vocabulario (k)', fontsize=12)
    plt.ylabel('Accuracy del Modelo (SVM)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(config.K_VALUES_TO_TEST)
    plt.tight_layout()

    best_k_index = np.argmax(accuracies)
    best_k = config.K_VALUES_TO_TEST[best_k_index]
    best_acc = accuracies[best_k_index]
    plt.annotate(f'Máximo: {best_acc:.4f}\nen k={best_k}', xy=(best_k, best_acc), xytext=(best_k, best_acc - 0.02 if best_acc > 0.5 else best_acc + 0.02),
                 arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle='arc3,rad=.2'),
                 ha='center', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    output_path = config.RESULTS_DIR / "impacto_de_k_accuracy.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")
    
    # --- Entrenamiento y Guardado del Mejor Modelo ---
    print("\n--- Entrenando el modelo final con la mejor configuración ---")
    print(f"Mejor valor de k encontrado: {best_k} (Accuracy: {best_acc:.4f})")
    
    print(f"Creando vocabulario final con k={best_k} usando todos los datos de entrenamiento...")
    final_training_descriptors = np.vstack([desc for i, desc in enumerate(all_descriptors) if i in train_indices and desc is not None])
    final_kmeans = MiniBatchKMeans(n_clusters=best_k, random_state=config.RANDOM_STATE, n_init='auto').fit(final_training_descriptors)

    print("Creando histogramas finales para el conjunto de entrenamiento...")
    X_train_final = np.array([crear_histograma(final_kmeans, all_descriptors[i], best_k) for i in train_indices])
    y_train_final = np.array([1 if labels[i] == 1 else 0 for i in train_indices])

    print("Entrenando clasificador SVM final...")
    final_svm = SVC(C=config.SVM_C, kernel=config.SVM_KERNEL, gamma=config.SVM_GAMMA, random_state=config.RANDOM_STATE, probability=True)
    final_svm.fit(X_train_final, y_train_final)

    model_components = {'kmeans_vocab': final_kmeans, 'svm_classifier': final_svm, 'k_value': best_k, 'descriptor_source': config.K_ANALYSIS_DESCRIPTORS_FILE}
    
    config.MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_filename = f"final_classification_model_{config.K_ANALYSIS_DESCRIPTORS_FILE.replace('.pkl', '')}_k{best_k}.pkl"
    model_path = config.MODELS_OUTPUT_DIR / model_filename
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_components, f)
        
    print(f"\n¡Modelo final guardado exitosamente en: {model_path}!")
    print("--- PASO 7 Completado ---")

if __name__ == "__main__":
    main()