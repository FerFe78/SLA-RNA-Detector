import joblib
import pickle
import os
import sys
from pathlib import Path

# Fix import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sla_detector import config

# Paths
INPUT_DIR = config.INTERMEDIATE_MODELS_MIXED_DIR
OUTPUT_FILE = INPUT_DIR / "SVM_SIFT_SIFT_local_eq_mixed.pkl"


def package_artifacts():
    print("Packaging artifacts...")
    
    try:
        # Load individual components
        # Note: joblib used for saving in train script
        kmeans = joblib.load(os.path.join(INPUT_DIR, "kmeans_model_mixed.pkl"))
        clf = joblib.load(os.path.join(INPUT_DIR, "svm_model_mixed.pkl"))
        
        print("Components loaded.")
        
        # Create dictionary
        model_components = {
            'classifier': clf,
            'kmeans_vocab': kmeans,
            'config': {
                'vocab_size': 100, # Hardcoded from train script
                'detector': 'SIFT',
                'descriptor': 'SIFT',
                'normalization': 'local',
                'equalization': 'eq'
            }
        }
        
        # Save as single pickle (standard pickle used in app logic)
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(model_components, f)
            
        print(f"Successfully packaged model to: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error packaging: {e}")

if __name__ == "__main__":
    package_artifacts()
