import pickle
from pathlib import Path

def parse_model_filename(filename):
    """
    Extracts pipeline parameters from the model filename.
    Format: Classifier_Detector_Descriptor_Norm_Eq.pkl
    """
    try:
        # Handle Path objects
        if hasattr(filename, 'name'):
            filename = filename.name
            
        parts = filename.replace('.pkl', '').split('_')
        # Ej: 'SVM_fast_brief_local_eq.pkl' -> ['SVM', 'fast', 'brief', 'local', 'eq']
        params = {
            'classifier': parts[0],
            'detector': parts[1].upper(),
            'descriptor': parts[2].upper(),
            'normalization': parts[3],
            # If parts has 5 elements, the last one is likely 'eq'
            'equalization': 'eq' if 'eq' in parts else 'none',
            'filename': filename
        }
        return params
    except IndexError:
        return None

def load_classifier(model_path):
    """
    Loads a .pkl classification model.
    Returns the dictionary of components.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
        
    with open(path, 'rb') as f:
        model_components = pickle.load(f)
        
    return model_components
