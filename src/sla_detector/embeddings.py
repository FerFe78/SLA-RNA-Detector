import torch
import fm
import streamlit as st
from . import config

# Singleton storage for the model
_RNA_FM_MODEL = None
_RNA_FM_ALPHABET = None

def load_rna_fm_model():
    """
    Loads the RNA-FM model and alphabet. 
    Uses a singleton pattern (or Streamlit cache if called from app) to avoid reloading.
    """
    global _RNA_FM_MODEL, _RNA_FM_ALPHABET
    
    if _RNA_FM_MODEL is None:
        print(f"Loading RNA-FM model from {config.RNAFM_MODEL_PATH}...")
        model, alphabet = fm.pretrained.rna_fm_t12(str(config.RNAFM_MODEL_PATH))
        model.to(config.DEVICE).eval()
        _RNA_FM_MODEL = model
        _RNA_FM_ALPHABET = alphabet
        print("Model loaded successfully.")
    
    return _RNA_FM_MODEL, _RNA_FM_ALPHABET

def get_batch_embeddings(sequences, batch_size=8):
    """
    Generates embeddings for a list of sequences using GPU batch processing.
    
    Args:
        sequences: List of tuples (id, sequence_string).
        batch_size: Number of sequences to process at once.
        
    Returns:
        Dictionary {id: embedding_tensor}
    """
    model, alphabet = load_rna_fm_model()
    batch_converter = alphabet.get_batch_converter()
    
    results = {}
    
    # Process in batches
    for i in range(0, len(sequences), batch_size):
        batch_data = sequences[i : i + batch_size]
        
        # Helper: Clean sequences (RNA-FM expects upper case U)
        cleaned_batch = []
        original_lengths = []
        for name, seq in batch_data:
            clean_seq = seq.upper().replace('T', 'U')
            cleaned_batch.append((name, clean_seq))
            original_lengths.append(len(clean_seq))
            
        labels, strs, tokens = batch_converter(cleaned_batch)
        tokens = tokens.to(config.DEVICE)
        
        with torch.no_grad():
            out = model(tokens, repr_layers=[12])
            
        # Extract embeddings
        # out['representations'][12] shape: [batch, max_len + 2, embed_dim] (includes start/end tokens)
        rep = out['representations'][12]
        
        for j, (name, _) in enumerate(cleaned_batch):
            # Extract relevant part: from index 1 to length+1
            length = original_lengths[j]
            emb = rep[j, 1 : length + 1, :].cpu()
            results[name] = emb
            
    return results

def get_single_embedding(sequence):
    """Convenience wrapper for a single sequence."""
    res = get_batch_embeddings([("query", sequence)], batch_size=1)
    return res["query"]
