# o_config.py

from pathlib import Path
import torch
import numpy as np

# --- Configuración Base ---
BASE_DIR = Path(__file__).resolve().parent.parent # code/scripts -> code/ -> BASE is code/
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Rutas de Entrada y Salida ---
DATA_DIR = BASE_DIR / "data"
RAW_CSV_PATH = DATA_DIR / "dengue_training_full.csv" # Not used by scripts directly usually but good ref
CSV_PATH = DATA_DIR / "dengue_training_full.csv"     # <--- TARGET INPUT FILE for 1_generar_embeddings.py

MODELS_INPUT_DIR = BASE_DIR / "models"
RNAFM_MODEL_PATH = MODELS_INPUT_DIR / "RNA-FM_pretrained.pth"

OUTPUT_DIR = BASE_DIR / "output" # Using the existing 'output' folder
RESULTS_DIR = OUTPUT_DIR / "results"
DESCRIPTORS_DIR = OUTPUT_DIR / "descriptors"
VOCAB_DIR = OUTPUT_DIR / "vocabularies"
HISTOGRAMS_DIR = OUTPUT_DIR / "histograms"

IMAGE_DIRS = {
    "global": OUTPUT_DIR / "images_global_norm",
    "local": OUTPUT_DIR / "images_local_norm",
    "global_eq": OUTPUT_DIR / "images_global_norm_eq",
    "local_eq": OUTPUT_DIR / "images_local_norm_eq",
}

INTERMEDIATE_MODELS_DIR = OUTPUT_DIR / "intermediate_models"
INTERMEDIATE_MODELS_MIXED_DIR = OUTPUT_DIR / "intermediate_models_mixed"
MODELS_OUTPUT_DIR = OUTPUT_DIR / "trained_models"

# --- Data Generation Params ---
FORZAR_REGENERACION = True # Generar todo desde cero para este nuevo dataset

random_state = 42 # Used by modules importing as config.random_state
RANDOM_STATE = 42

# --- Model/Training Params ---
VOCAB_SIZE = 500
TEST_SIZE = 0.20

RF_N_ESTIMATORS = 400 
SVM_C = 10
SVM_KERNEL = 'rbf' 
SVM_GAMMA = 'scale'

# Validation configs
VALIDATION_DESCRIPTORS_FILE = "fast_local_eq.pkl" # Default descriptor scheme to validate
VALIDATION_MODEL_FILE = "SVM_fast_brief_local_eq.pkl"
CV_FOLDS = 5
PARAM_GRID_SVM = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf']
}
LEARNING_CURVE_TRAIN_SIZES = np.linspace(0.1, 1.0, 10)

K_ANALYSIS_DESCRIPTORS_FILE = "fast_brief_local_eq.pkl"
K_VALUES_TO_TEST = [25, 50, 100, 150, 200, 250, 300, 400, 500]
