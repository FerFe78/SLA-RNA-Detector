import pandas    
import torch
import fm
import cv2
import skimage
import sklearn
import matplotlib
import streamlit
import numpy as np

print("--------------------------------------------------")
print("VERIFICATION SUCCESSFUL: All modules imported.")
print(f"Torch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print("--------------------------------------------------")
