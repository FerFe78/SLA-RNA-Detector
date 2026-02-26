import numpy as np
from skimage import exposure

def process_embedding_to_image(embedding):
    """
    Applies Local Normalization and Histogram Equalization to an embedding tensor.
    
    Args:
        embedding: Tensor or component of shape (L, D).
        
    Returns:
        img_eq_8bit: Equalized image (uint8) used for feature extraction.
        img_norm: Normalized image (uint8) for visualization.
    """
    local_min, local_max = embedding.min(), embedding.max()
    
    # Local Normalization (0-255)
    if local_max > local_min:
        img_array = ((embedding - local_min) / (local_max - local_min) * 255.0)
        
        # Handle tensor vs numpy
        if hasattr(img_array, 'byte'):
             img_array = img_array.byte().cpu().numpy()
        else:
             img_array = img_array.astype(np.uint8)
    else:
        # Handle zero-variance case
        shape = embedding.shape if hasattr(embedding, 'shape') else (1,1)
        img_array = np.zeros(shape, dtype=np.uint8)
        
    # Histogram Equalization
    img_eq_float = exposure.equalize_hist(img_array)
    img_eq_8bit = (img_eq_float * 255).astype(np.uint8)
    
    return img_eq_8bit, img_array
