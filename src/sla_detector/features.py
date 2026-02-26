import cv2
import numpy as np
import streamlit as st

def get_detector_and_extractor(detector_name, descriptor_name):
    """Factory for OpenCV detectors and extractors."""
    try:
        # Mapeo de nombres a objetos de detectores
        detectors = {
            'FAST': cv2.FastFeatureDetector_create(),
            'SIFT': cv2.SIFT_create(),
            'ORB': cv2.ORB_create(),
            'BRISK': cv2.BRISK_create()
        }

        # Mapeo de nombres a objetos de extractores
        extractors = {
            'BRIEF': cv2.xfeatures2d.BriefDescriptorExtractor_create(),
            'SIFT': cv2.SIFT_create(),
            'ORB': cv2.ORB_create(),
            'BRISK': cv2.BRISK_create()
        }
        
        # Optimization: use same object if detector == extractor
        if detector_name == descriptor_name and detector_name in ['SIFT', 'ORB', 'BRISK']:
             return detectors[detector_name], detectors[detector_name]
        
        if detector_name not in detectors:
            raise ValueError(f"Detector '{detector_name}' no reconocido.")
        if descriptor_name not in extractors:
            raise ValueError(f"Descriptor '{descriptor_name}' no reconocido.")
            
        return detectors[detector_name], extractors[descriptor_name]

    except AttributeError as e:
        raise ImportError(f"Error initializing OpenCV components: {e}. Check opencv-contrib-python installation.")

def extract_features(image, detector_name, descriptor_name):
    """
    Extracts keypoints and descriptors from an image.
    Returns descriptors as np.float32.
    """
    detector, extractor = get_detector_and_extractor(detector_name.upper(), descriptor_name.upper())
    
    keypoints = detector.detect(image, None)
    keypoints, descriptors = extractor.compute(image, keypoints)
    
    if descriptors is not None:
        descriptors = np.float32(descriptors)
        
    return descriptors, keypoints

def create_bow_histogram(kmeans_model, descriptors, vocab_size):
    """
    Creates a Bag-of-Visual-Words histogram.
    """
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocab_size)

    # 1. Match dtype to model
    required_dtype = kmeans_model.cluster_centers_.dtype
    descriptores_adaptados = descriptors.astype(required_dtype)

    # 2. Predict visual words
    palabras_visuales = kmeans_model.predict(descriptores_adaptados)
    
    # 3. Histogram
    hist, _ = np.histogram(palabras_visuales, bins=np.arange(vocab_size + 1))
    hist = hist.astype(float)
    
    # 4. Normalize
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
        
    return hist.reshape(1, -1)
