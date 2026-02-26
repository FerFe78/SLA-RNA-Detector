# o_config.py

from pathlib import Path
import torch
import numpy as np # Necesario para la curva de aprendizaje

# --- Configuración Base ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Subir 3 niveles para llegar a la raíz (src/sla_detector/config.py -> Proyecto_Integrador_IA)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Rutas de Entrada y Salida (Completas) ---
DATA_DIR = BASE_DIR / "data"
RAW_CSV_PATH = DATA_DIR / "base_de_datos_SLA.csv"
CSV_PATH = DATA_DIR / "SLA_limpio.csv"
MODELS_INPUT_DIR = BASE_DIR / "src" / "models"  # Ajustado si los modelos quedaron en src/models
# Si moviste los modelos a otro lado, ajustamos aquí.
RNAFM_MODEL_PATH = MODELS_INPUT_DIR / "RNA-FM_pretrained.pth"

OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "results"
DESCRIPTORS_DIR = OUTPUT_DIR / "descriptors" # <-- Línea que faltaba
VOCAB_DIR = OUTPUT_DIR / "vocabularies"
HISTOGRAMS_DIR = OUTPUT_DIR / "histograms"
IMAGE_DIRS = {
    # Imágenes originales (usadas en 2_crear_imagenes.py)
    "global": OUTPUT_DIR / "images_global_norm",
    "local": OUTPUT_DIR / "images_local_norm",
    # Imágenes ecualizadas (usadas en 3_ecualizar_imagenes.py y en adelante)
    "global_eq": OUTPUT_DIR / "images_global_norm_eq",
    "local_eq": OUTPUT_DIR / "images_local_norm_eq",
}
INTERMEDIATE_MODELS_DIR = OUTPUT_DIR / "intermediate_models"
INTERMEDIATE_MODELS_MIXED_DIR = OUTPUT_DIR / "intermediate_models_mixed"
MODELS_OUTPUT_DIR = OUTPUT_DIR / "trained_models"

# ===================================================================
# --- PARÁMETROS DEL PIPELINE PRINCIPAL Y ANÁLISIS ---
# ===================================================================

FORZAR_REGENERACION = False
RANDOM_STATE = 42
VOCAB_SIZE = 500      # Tamaño del vocabulario para la ejecución principal y validación
TEST_SIZE = 0.20      # Proporción para el split 80/20

# Parámetros de los Clasificadores Clásicos
RF_N_ESTIMATORS = 400 
SVM_C = 10
SVM_KERNEL = 'rbf' 
SVM_GAMMA = 'scale'

# --- Configuración para 6_validar_overfitting.py ---
VALIDATION_DESCRIPTORS_FILE = "fast_local_eq.pkl"
VALIDATION_MODEL_FILE = "SVM_fast_brief_local_eq.pkl"
CV_FOLDS = 5
PARAM_GRID_SVM = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf']
}
# Usamos porcentajes, el script robusto lo manejará
LEARNING_CURVE_TRAIN_SIZES = np.linspace(0.1, 1.0, 10)

# --- Configuración para 7_analizar_impacto_k.py ---
K_ANALYSIS_DESCRIPTORS_FILE = "fast_brief_local_eq.pkl"
K_VALUES_TO_TEST = [25, 50, 100, 150, 200, 250, 300, 400, 500]