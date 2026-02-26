import pandas as pd
import numpy as np
import os
import sys
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sla_detector import config, embeddings, features, image_processing

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent # code/
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = config.INTERMEDIATE_MODELS_MIXED_DIR

# Data
ORIGINAL_DATA_PATH = DATA_DIR / "SLA_limpio.csv" 
ARTIFICIAL_DATA_PATH = DATA_DIR / "base_de_datos_artificiales.csv"

CONFIGURATIONS = [
    ('BRISK', 'BRISK'),
    ('FAST', 'BRIEF'),
    ('FAST', 'BRISK'),
    ('FAST', 'SIFT'),
    ('ORB', 'ORB'),
    ('SIFT', 'SIFT')
]

def load_and_merge_data():
    # Load Original
    try:
        df_real = pd.read_csv(ORIGINAL_DATA_PATH)
        if 'secuencia' not in df_real.columns: df_real = pd.read_csv(ORIGINAL_DATA_PATH, sep=';')
    except:
        df_real = pd.read_csv(ORIGINAL_DATA_PATH, sep=';', encoding='latin1')
        
    df_real.columns = [c.strip().lower() for c in df_real.columns]
    if 'nombre' in df_real.columns: df_real = df_real.rename(columns={'nombre': 'id'})
    if 'accession' in df_real.columns: df_real = df_real.rename(columns={'accession': 'id'})
    if 'sla' in df_real.columns: df_real = df_real.rename(columns={'sla': 'label'})
    
    # Load Artificial
    df_fake = pd.read_csv(ARTIFICIAL_DATA_PATH, sep=';', encoding='latin1')
    df_fake.columns = [c.strip().lower() for c in df_fake.columns]
    df_fake = df_fake.rename(columns={'nombre': 'id', 'sla': 'label'})
    
    def clean_label(val): 
        return 1 if str(val).strip() == '+' or str(val) == '1' else 0
        
    if df_real.get('label') is not None and df_real['label'].dtype == object: 
        df_real['label'] = df_real['label'].apply(clean_label)
    df_fake['label'] = df_fake['label'].apply(clean_label)
    
    # Merge
    return pd.concat([df_real, df_fake], ignore_index=True)

def evaluate_all():
    print("Loading Data...")
    df = load_and_merge_data()
    
    sequences = [(str(row['id']), str(row['secuencia'])) for _, row in df.iterrows()]
    labels = df['label'].values
    
    print(f"Generating embeddings for {len(sequences)} sequences...")
    cache_path = DATA_DIR / "embeddings_cache_mixed.pkl"
    
    if cache_path.exists():
        print("Loading embeddings from cache...")
        with open(cache_path, 'rb') as f:
            emb_dict = pickle.load(f)
    else:
        emb_dict = embeddings.get_batch_embeddings(sequences, batch_size=8)
        with open(cache_path, 'wb') as f:
            pickle.dump(emb_dict, f)
    
    results_summary = []
    
    for detector_name, descriptor_name in CONFIGURATIONS:
        model_filename = f"svm_{detector_name.lower()}_{descriptor_name.lower()}_local_eq_mixed.pkl"
        model_path = MODELS_DIR / model_filename
        
        if not model_path.exists():
            print(f"Skipping {model_filename}: File not found.")
            continue
            
        print(f"\nEvaluating {detector_name} + {descriptor_name}...")
        
        # Load Model Components
        try:
            with open(model_path, 'rb') as f:
                components = pickle.load(f)
            clf = components['classifier']
            kmeans = components['kmeans_vocab']
            k = components['config']['vocab_size']
        except Exception as e:
            print(f"Error loading {model_filename}: {e}")
            continue

        # Extract Features (Same split logic requires same features)
        X_hist = []
        valid_y = [] # Store labels for successful extractions
        
        for i, (name, _) in enumerate(sequences):
            if name not in emb_dict: continue
            try:
                emb = emb_dict[name]
                img_eq, _ = image_processing.process_embedding_to_image(emb)
                des, _ = features.extract_features(img_eq, detector_name, descriptor_name)
                
                if des is not None:
                    hist = features.create_bow_histogram(kmeans, des, k)
                    X_hist.append(hist[0])
                    valid_y.append(labels[i])
            except: pass
            
        X_hist = np.array(X_hist)
        y = np.array(valid_y)
        
        if len(X_hist) == 0:
            print("No features extracted.")
            continue
            
        # Recreate Split (Validation Set)
        # Random State 42 matches training script
        _, X_test, _, y_test = train_test_split(X_hist, y, test_size=0.2, random_state=42, stratify=y)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        results_summary.append({
            "Model": f"{detector_name}+{descriptor_name}",
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })
        print(f"-> Accuracy: {acc:.4f}")

    # Print Summary Table
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10}")
    print("-" * 70)
    for res in results_summary:
        print(f"{res['Model']:<20} | {res['Accuracy']:.4f}     | {res['Precision']:.4f}    | {res['Recall']:.4f}     | {res['F1 Score']:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    evaluate_all()
