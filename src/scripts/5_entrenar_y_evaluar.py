# 5_entrenar_y_evaluar.py

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import o_config as config

# --- Función Auxiliar ---
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

def main():
    print("--- PASO 5: Entrenamiento, Evaluación y Guardado de Modelos ---")
    
    descriptor_files = list(config.DESCRIPTORS_DIR.glob('*.pkl'))
    if not descriptor_files:
        print(f"ERROR: No se encontraron archivos de descriptores en '{config.DESCRIPTORS_DIR}'.")
        return

    classifiers = {
        'SVM': SVC(C=config.SVM_C, kernel=config.SVM_KERNEL, gamma=config.SVM_GAMMA, random_state=config.RANDOM_STATE, probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=config.RF_N_ESTIMATORS, random_state=config.RANDOM_STATE)
    }
    
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.INTERMEDIATE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for desc_path in descriptor_files:
        print(f"\n{'='*60}\nProcesando: {desc_path.name}\n{'='*60}")
        
        with open(desc_path, 'rb') as f:
            data = pickle.load(f)
        all_descriptors, labels = data['features'], data['labels']
        
        # 1. División de Datos
        indices = np.arange(len(all_descriptors))
        train_indices, test_indices, _, _ = train_test_split(
            indices, labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=labels
        )
        print(f"Datos divididos: {len(train_indices)} para entrenamiento, {len(test_indices)} para prueba.")

        # 2. Creación del Vocabulario (SOLO con datos de entrenamiento)
        print(f"Creando vocabulario k={config.VOCAB_SIZE}...")
        training_descriptors = [all_descriptors[i] for i in train_indices if all_descriptors[i] is not None]
        if not training_descriptors:
            print("No hay descriptores en el conjunto de entrenamiento. Saltando.")
            continue
        
        kmeans = MiniBatchKMeans(
            n_clusters=config.VOCAB_SIZE,
            random_state=config.RANDOM_STATE,
            n_init='auto'
        ).fit(np.vstack(training_descriptors))

        # 3. Creación de Histogramas
        print("Creando histogramas para todo el conjunto de datos...")
        X_hist = np.array([crear_histograma(kmeans, desc, config.VOCAB_SIZE) for desc in all_descriptors])
        y_hist = np.array([1 if label == 1 else 0 for label in labels])

        X_train, X_test = X_hist[train_indices], X_hist[test_indices]
        y_train, y_test = y_hist[train_indices], y_hist[test_indices]

        # 4. Entrenamiento, Evaluación y Guardado
        for clf_name, clf in classifiers.items():
            print(f"\n--- Entrenando y Evaluando: {clf_name} ---")
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['No-SLA (0)', 'SLA (1)'])
            
            print(f"Accuracy: {accuracy:.4f}")
            print("Informe de Clasificación:")
            print(report)

            # Guardar matriz de confusión
            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax, cmap=plt.cm.Blues, values_format='d')
            ax.set_title(f"Matriz de Confusión - {clf_name}\n({desc_path.stem})")
            result_img_path = config.RESULTS_DIR / f"cm_{clf_name}_{desc_path.stem}.png"
            plt.savefig(result_img_path)
            plt.close(fig)
            
            # Guardar el modelo entrenado
            model_components = {
                'kmeans_vocab': kmeans,
                'classifier': clf,
                'config': {
                    'descriptor_file': desc_path.name,
                    'vocab_size': config.VOCAB_SIZE,
                    'classifier_name': clf_name
                }
            }
            model_filename = f"{clf_name}_{desc_path.stem}.pkl"
            model_path = config.INTERMEDIATE_MODELS_DIR / model_filename
            with open(model_path, 'wb') as f:
                pickle.dump(model_components, f)
            print(f"Modelo guardado en: {model_path}")

    print("\n--- PASO 5 Completado ---")

if __name__ == "__main__":
    main()