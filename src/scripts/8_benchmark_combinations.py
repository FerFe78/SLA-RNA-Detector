import cv2
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import os

# Configuración de Rutas
BASE_DIR = Path(r"C:\Users\lopez\Documents\agtest\Proyecto_Integrador_IA\code\output")
IMAGE_DIR = BASE_DIR / "images_local_norm_eq"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
VOCAB_SIZE = 500
TEST_SIZE = 0.20

def create_histogram(kmeans_model, descriptors, vocab_size):
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocab_size)
    palabras_visuales = kmeans_model.predict(descriptors)
    hist, _ = np.histogram(palabras_visuales, bins=np.arange(vocab_size + 1))
    hist = hist.astype(float)
    if np.sum(hist) > 0:
        hist /= np.linalg.norm(hist)
    return hist

def main():
    print("--- BENCHMARK: Comparación de Detectores y Descriptores ---")
    
    combinations = [
        ('FAST', 'BRIEF'),
        ('FAST', 'SIFT'),
        ('FAST', 'BRISK'),
        ('SIFT', 'SIFT'),
        ('BRISK', 'BRISK'),
        ('ORB', 'ORB')
    ]

    # Pre-cargar imágenes y etiquetas
    image_paths = sorted(list(IMAGE_DIR.glob('*.png')))
    print(f"Cargando {len(image_paths)} imágenes...")
    images = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in image_paths]
    labels = [int(p.stem.split('-')[1]) for p in image_paths]

    results = []

    for det_name, desc_name in combinations:
        print(f"\nEvaluando: {det_name} + {desc_name}")
        
        # 1. Inicializar componentes
        if det_name == 'FAST':
            detector = cv2.FastFeatureDetector_create()
        elif det_name == 'SIFT':
            detector = cv2.SIFT_create()
        elif det_name == 'BRISK':
            detector = cv2.BRISK_create()
        elif det_name == 'ORB':
            detector = cv2.ORB_create(nfeatures=2000)
            
        if desc_name == 'BRIEF':
            extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        elif desc_name == 'SIFT':
            extractor = cv2.SIFT_create()
        elif desc_name == 'BRISK':
            extractor = cv2.BRISK_create()
        elif desc_name == 'ORB':
            extractor = cv2.ORB_create(nfeatures=2000)

        # 2. Benchmark de extracción de características
        all_descriptors = []
        valid_indices = []
        start_time = time.time()
        
        for i, img in enumerate(images):
            if det_name == desc_name and det_name in ['SIFT', 'BRISK', 'ORB']:
                kp, des = detector.detectAndCompute(img, None)
            else:
                kp = detector.detect(img, None)
                kp, des = extractor.compute(img, kp)
            
            if des is not None and len(des) > 0:
                all_descriptors.append(np.float32(des))
                valid_indices.append(i)
        
        total_time = time.time() - start_time
        avg_time = (total_time / len(images)) * 1000 # ms por imagen
        
        print(f"  - Tiempo medio: {avg_time:.2f} ms/imagen")

        # 3. Filtrar etiquetas para imágenes con descriptores
        current_labels = [labels[i] for i in valid_indices]
        
        # 4. Train/Test split
        idx_train, idx_test = train_test_split(
            np.arange(len(all_descriptors)), 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=current_labels
        )
        
        # 5. BoVW: K-Means
        training_des = np.vstack([all_descriptors[i] for i in idx_train])
        kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, random_state=RANDOM_STATE, n_init='auto').fit(training_des)
        
        # 6. Crear Histogramas
        X = np.array([create_histogram(kmeans, d, VOCAB_SIZE) for d in all_descriptors])
        y = np.array(current_labels)
        
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]
        
        # 7. Clasificación SVM
        clf = SVC(C=10, kernel='rbf', random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  - Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        results.append({
            'Combination': f"{det_name}+{desc_name}",
            'Accuracy': acc,
            'F1-Score': f1,
            'Time_ms': avg_time
        })

    # Guardar resultados
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "benchmark_comparison.csv", index=False)
    
    # Generar Gráfico
    plt.figure(figsize=(12, 7))
    plt.style.use('dark_background')
    
    # Scatter plot: Tiempo vs Rendimiento
    for i, row in df.iterrows():
        plt.scatter(row['Time_ms'], row['Accuracy'], s=200, label=row['Combination'], alpha=0.7)
        plt.text(row['Time_ms']+0.5, row['Accuracy'], row['Combination'], fontsize=9, verticalalignment='center')

    plt.title("Rendimiento vs Eficiencia de Combinaciones Visitales", fontsize=15, color='teal')
    plt.xlabel("Tiempo Medio (ms / imagen)", fontsize=12)
    plt.ylabel("Precisión (Accuracy)", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "benchmark_efficiency.png")
    print(f"\nBenchmark completado. Gráfico guardado en: {RESULTS_DIR / 'benchmark_efficiency.png'}")

if __name__ == "__main__":
    main()
