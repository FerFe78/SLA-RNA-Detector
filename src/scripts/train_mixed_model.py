import pandas as pd
import os
import torch
import joblib
import numpy as np
import cv2
import sys
import pickle
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Fix path to find 'sla_detector' package (two levels up from scripts/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sla_detector import config, embeddings, image_processing, features

# Paths (Using config where possible, or relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent # code/
DATA_DIR = PROJECT_ROOT / "data"

# User requested using 'SLA_limpio.csv' which is already clean
ORIGINAL_DATA_PATH = DATA_DIR / "SLA_limpio.csv" 
ARTIFICIAL_DATA_PATH = DATA_DIR / "base_de_datos_artificiales.csv"

# Output inside code/output/...
OUTPUT_DIR = config.INTERMEDIATE_MODELS_MIXED_DIR

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurations to train
# (Detector, Descriptor)
CONFIGURATIONS = [
    ('BRISK', 'BRISK'),
    ('FAST', 'BRIEF'),
    ('FAST', 'BRISK'),
    ('FAST', 'SIFT'),
    ('ORB', 'ORB'),
    ('SIFT', 'SIFT')
]

def load_and_merge_data():
    print("Loading datasets...")
    # Load Original
    try:
        df_real = pd.read_csv(ORIGINAL_DATA_PATH)
        if 'secuencia' not in df_real.columns: 
             df_real = pd.read_csv(ORIGINAL_DATA_PATH, sep=';')
    except:
        df_real = pd.read_csv(ORIGINAL_DATA_PATH, sep=';', encoding='latin1')
        
    # Normalize Real
    df_real.columns = [c.strip().lower() for c in df_real.columns]
    if 'nombre' in df_real.columns: df_real = df_real.rename(columns={'nombre': 'id'})
    if 'accession' in df_real.columns: df_real = df_real.rename(columns={'accession': 'id'})
    if 'sla' in df_real.columns: df_real = df_real.rename(columns={'sla': 'label'})
    
    # Load Artificial
    df_fake = pd.read_csv(ARTIFICIAL_DATA_PATH, sep=';', encoding='latin1')
    df_fake.columns = [c.strip().lower() for c in df_fake.columns]
    df_fake = df_fake.rename(columns={'nombre': 'id', 'sla': 'label'})
    
    # Standardize labels (+/- to 1/0)
    def clean_label(val):
        if str(val).strip() == '+' or str(val) == '1': return 1
        return 0
        
    if df_real.get('label') is not None and df_real['label'].dtype == object:
        df_real['label'] = df_real['label'].apply(clean_label)
        
    df_fake['label'] = df_fake['label'].apply(clean_label)
    
    # Select columns
    df_real = df_real[['id', 'secuencia', 'label']]
    df_fake = df_fake[['id', 'secuencia', 'label']]
    
    # Merge
    df_combined = pd.concat([df_real, df_fake], ignore_index=True)
    print(f"Merged Data: {len(df_real)} real + {len(df_fake)} artificial = {len(df_combined)} total.")
    return df_combined

def perform_training_cycle(df_combined, emb_dict, detector_name, descriptor_name):
    print(f"\n{'='*60}\nProcessing Configuration: {detector_name} + {descriptor_name}\n{'='*60}")
    
    sequences = [(str(row['id']), str(row['secuencia'])) for _, row in df_combined.iterrows()]
    labels = df_combined['label'].values
    
    # 1. Feature Extraction
    print(f"Extracting features using {detector_name}/{descriptor_name}...")
    dataset_descriptors = []
    valid_labels = []
    
    # Initialize detector/extractor once
    # We use the factory in features.py manually or just call extract_features per image
    # Calling per image is safer for the existing pipeline logic
    
    for i, (name, _) in enumerate(sequences):
        if name not in emb_dict: continue
            
        emb = emb_dict[name]
        try:
            img_eq, _ = image_processing.process_embedding_to_image(emb)
            # Use the feature extraction module
            # Note: extract_features returns (descriptors, keypoints)
            des, _ = features.extract_features(img_eq, detector_name, descriptor_name)
            
            if des is not None:
                dataset_descriptors.append(des)
                valid_labels.append(labels[i])
        except Exception as e:
            # print(f"Error processing {name}: {e}") # Reduce verbsity
            pass

    if not dataset_descriptors:
        print(f"Skipping {detector_name}_{descriptor_name}: No features extracted.")
        return

    # 2. KMeans (BoW)
    print("Training KMeans Vocabulary...")
    all_descriptors = np.vstack(dataset_descriptors)
    k_value = 100 # Standard Vocab Size
    
    kmeans = MiniBatchKMeans(n_clusters=k_value, random_state=42, batch_size=1000, n_init='auto')
    kmeans.fit(all_descriptors)
    
    # 3. Create Histograms
    print("Creating Histograms...")
    X_hist = []
    for des in dataset_descriptors:
        hist = features.create_bow_histogram(kmeans, des, k_value)
        X_hist.append(hist[0]) 
    
    X_hist = np.array(X_hist)
    y = np.array(valid_labels)
    
    # 4. Train SVM
    print("Training SVM Classifier...")
    # Validation split for metrics
    X_train, X_test, y_train, y_test = train_test_split(X_hist, y, test_size=0.2, random_state=42, stratify=y)
    
    clf_val = SVC(kernel='linear', probability=True, random_state=42)
    clf_val.fit(X_train, y_train)
    y_pred = clf_val.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation Accuracy: {acc:.4f}")
    
    # Full Retrain
    clf_full = SVC(kernel='linear', probability=True, random_state=42)
    clf_full.fit(X_hist, y)
    
    # 5. Save Artifact (Directly in format for App)
    output_filename = f"svm_{detector_name.lower()}_{descriptor_name.lower()}_local_eq_mixed.pkl"
    output_path = OUTPUT_DIR / output_filename
    
    model_components = {
        'classifier': clf_full,
        'kmeans_vocab': kmeans,
        'config': {
            'vocab_size': k_value,
            'detector': detector_name,
            'descriptor': descriptor_name,
            'normalization': 'local',
            'equalization': 'eq',
            'mixed_training': True
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_components, f)
        
    print(f"Saved model to: {output_path}")

def train_all_models():
    # Load Data Once
    df = load_and_merge_data()
    
    # Pre-compute Embeddings Once (Expensive step)
    sequences = [(str(row['id']), str(row['secuencia'])) for _, row in df.iterrows()]
    print(f"Generating embeddings for {len(sequences)} sequences (Unified)...")
    emb_dict = embeddings.get_batch_embeddings(sequences, batch_size=8)
    
    # Iterate Configurations
    for detector, descriptor in CONFIGURATIONS:
        try:
            perform_training_cycle(df, emb_dict, detector, descriptor)
        except Exception as e:
            print(f"CRITICAL ERROR training {detector}+{descriptor}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Starting Multi-Model Mixed Training Pipeline...")
    train_all_models()
    print("\nAll training cycles completed.")
