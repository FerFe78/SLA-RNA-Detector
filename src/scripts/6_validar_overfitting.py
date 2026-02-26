# 6_validar_overfitting_refactorizado.py

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # NUEVO: Importamos pandas para mostrar los resultados de GridSearchCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import o_config as config

# --- Función Auxiliar ---
def crear_histograma(kmeans, descriptores, k):
    """Crea un histograma BoVW para una imagen."""
    if descriptores is None or len(descriptores) == 0:
        return np.zeros(k)
    palabras = kmeans.predict(descriptores)
    hist, _ = np.histogram(palabras, bins=np.arange(k + 1))
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= np.linalg.norm(hist)
    return hist

def plot_learning_curve_robust(estimator, title, X, y, cv, train_sizes_frac):
    """Genera el gráfico de la curva de aprendizaje con muestreo manual robusto."""
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Ejemplos de entrenamiento")
    plt.ylabel("Score (Accuracy)")
    plt.grid()

    n_train_samples = int(len(X) * (cv.get_n_splits() - 1) / cv.get_n_splits())
    train_sizes_abs = (train_sizes_frac * n_train_samples).astype(int)
    train_sizes_abs = np.unique(np.maximum(train_sizes_abs, 2))

    train_scores_mean, test_scores_mean = [], []

    for n_samples in tqdm(train_sizes_abs, desc="Calculando Curva de Aprendizaje"):
        fold_train_scores, fold_test_scores = [], []
        for train_idx, test_idx in cv.split(X, y):
            X_train_full, y_train_full = X[train_idx], y[train_idx]
            
            np.random.seed(config.RANDOM_STATE)
            sub_indices = np.random.choice(np.arange(len(X_train_full)), size=n_samples, replace=False)
            X_train_sub, y_train_sub = X_train_full[sub_indices], y_train_full[sub_indices]

            if len(np.unique(y_train_sub)) < 2:
                continue

            estimator.fit(X_train_sub, y_train_sub)
            fold_train_scores.append(estimator.score(X_train_sub, y_train_sub))
            fold_test_scores.append(estimator.score(X[test_idx], y[test_idx]))

        if fold_train_scores:
            train_scores_mean.append(np.mean(fold_train_scores))
            test_scores_mean.append(np.mean(fold_test_scores))

    plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Score de Entrenamiento")
    plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g", label="Score de Validación Cruzada")
    plt.legend(loc="best")
    plt.ylim(0.8, 1.01)

    lc_path = config.RESULTS_DIR / f"learning_curve_{config.VALIDATION_MODEL_FILE.replace('.pkl', '')}.png"
    plt.savefig(lc_path)
    plt.close()
    print(f"Gráfico de curvas de aprendizaje guardado en: {lc_path}")

def main():
    print("--- PASO 6: Validación de Sobreajuste (Cargando Modelo) ---")
    
    model_path = config.INTERMEDIATE_MODELS_DIR / config.VALIDATION_MODEL_FILE
    if not model_path.exists():
        print(f"ERROR: No se encontró el archivo del modelo '{model_path}'.")
        print("Ejecuta '5_entrenar_y_evaluar.py' y configura 'VALIDATION_MODEL_FILE' en o_config.py.")
        return
        
    print(f"Cargando modelo y datos desde: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    clf = model_data['classifier']
    model_config = model_data['config']
    
    desc_path = config.DESCRIPTORS_DIR / model_config['descriptor_file']
    with open(desc_path, 'rb') as f:
        data = pickle.load(f)
    all_descriptors, labels = data['features'], data['labels']
    
    print("Recreando histogramas con el vocabulario del modelo...")
    kmeans = model_data['kmeans_vocab']
    X_hist = np.array([crear_histograma(kmeans, desc, model_config['vocab_size']) for desc in all_descriptors])
    y_hist = np.array([1 if label == 1 else 0 for label in labels])
    
    cv_strategy = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    
    # --- 1. Validación Cruzada ---
    print("\n--- 1. Ejecutando Validación Cruzada Detallada ---")
    accuracies = []
    # NUEVO: Se añade enumerate para mostrar el número de fold
    for i, (train_index, test_index) in enumerate(tqdm(cv_strategy.split(X_hist, y_hist), total=config.CV_FOLDS, desc="Validación Cruzada")):
        model = clf.fit(X_hist[train_index], y_hist[train_index])
        score = accuracy_score(y_hist[test_index], model.predict(X_hist[test_index]))
        accuracies.append(score)
        # NUEVO: Imprimir el resultado de cada fold
        print(f"  Fold {i+1}/{config.CV_FOLDS} -> Accuracy: {score:.4f}")
    
    print("\nResumen de Validación Cruzada:")
    print(f"Scores individuales: {[f'{acc:.4f}' for acc in accuracies]}")
    print(f"Accuracy Promedio: {np.mean(accuracies):.4f} \u00B1 {np.std(accuracies):.4f}")
    
    # --- 2. Curvas de Aprendizaje ---
    print("\n--- 2. Generando Curvas de Aprendizaje ---")
    plot_learning_curve_robust(
        clf, 
        f"Curvas de Aprendizaje\n({config.VALIDATION_MODEL_FILE})", 
        X_hist, y_hist, 
        cv=cv_strategy, 
        train_sizes_frac=config.LEARNING_CURVE_TRAIN_SIZES
    )

    # --- 3. Búsqueda de Hiperparámetros con GridSearchCV ---
    print("\n--- 3. Ejecutando GridSearchCV ---")
    if isinstance(clf, SVC):
        # El parámetro 'refit=True' (valor por defecto) es importante, ya que reentrena el mejor
        # estimador en todo el conjunto de datos al final.
        grid_search = GridSearchCV(
            SVC(random_state=config.RANDOM_STATE), 
            param_grid=config.PARAM_GRID_SVM, 
            cv=cv_strategy, 
            scoring='accuracy', 
            n_jobs=-1, 
            verbose=1
        )
        grid_search.fit(X_hist, y_hist)
        
        # NUEVO: Mostrar todos los resultados de GridSearchCV en una tabla
        print("\n--- Resultados Completos de GridSearchCV ---")
        results_df = pd.DataFrame(grid_search.cv_results_)
        
        # Seleccionar y ordenar las columnas más relevantes para la visualización
        param_cols = [col for col in results_df.columns if col.startswith('param_')]
        metric_cols = ['mean_test_score', 'std_test_score', 'rank_test_score']
        
        # Para evitar el SettingWithCopyWarning, usamos .copy()
        results_display = results_df[param_cols + metric_cols].copy()
        results_display.sort_values(by='rank_test_score', inplace=True)
        
        # Ajustar opciones de visualización de pandas
        pd.set_option('display.max_rows', None) # Mostrar todas las filas
        pd.set_option('display.max_columns', None) # Mostrar todas las columnas
        pd.set_option('display.width', 120) # Ancho de la tabla en la consola

        print(results_display)
        
        print("\n--- Resumen de GridSearchCV ---")
        print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
        print(f"Mejor score de validación cruzada: {grid_search.best_score_:.4f}")
        
    else:
        print("GridSearchCV no se ejecuta para modelos que no son SVM en esta configuración.")

    print("\n--- PASO 6 Completado ---")

if __name__ == "__main__":
    main()