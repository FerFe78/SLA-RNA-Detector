# 3_ecualizar_imagenes.py
import os
from pathlib import Path
from PIL import Image
import numpy as np
from skimage import exposure
from tqdm import tqdm

import o_config as config

def ecualizar_y_guardar(imagen_path: Path, dir_salida: Path):
    """
    Carga una imagen, aplica ecualización de histograma y la guarda.
    """
    try:
        # Cargar la imagen en escala de grises
        img = Image.open(imagen_path).convert('L')
        img_array = np.array(img)
        
        # Aplicar ecualización de histograma
        # exposure.equalize_hist devuelve una imagen en punto flotante en [0, 1]
        img_eq_float = exposure.equalize_hist(img_array)
        
        # Convertir de nuevo a 8-bit [0, 255]
        img_eq_8bit = (img_eq_float * 255).astype(np.uint8)
        
        # Crear la nueva imagen y guardarla
        img_final = Image.fromarray(img_eq_8bit, 'L')
        img_final.save(dir_salida / imagen_path.name)
        
    except Exception as e:
        print(f"Error procesando {imagen_path}: {e}")

def main():
    print("--- PASO 3: Ecualización de Histograma para Imágenes en Escala de Grises ---")

    # Estrategias en escala de grises a las que se les aplicará ecualización
    estrategias_fuente = ["global", "local"]
    
    for estrategia in estrategias_fuente:
        dir_fuente = config.IMAGE_DIRS[estrategia]
        # La clave para el directorio de salida será la + '_eq'
        dir_salida = config.IMAGE_DIRS[f"{estrategia}_eq"]

        print(f"\nProcesando estrategia: '{estrategia}'")
        print(f"  - Directorio fuente: {dir_fuente}")
        print(f"  - Directorio salida: {dir_salida}")

        if not dir_fuente.exists() or not any(dir_fuente.iterdir()):
            print(f"  - ADVERTENCIA: El directorio fuente está vacío o no existe. Saltando esta estrategia.")
            print(f"  - Asegúrate de haber ejecutado '3_crear_imagenes_variables.py' primero.")
            continue

        # Crear directorio de salida si no existe
        dir_salida.mkdir(parents=True, exist_ok=True)
        
        # Comprobar si se debe forzar la regeneración
        if any(dir_salida.iterdir()) and not config.FORZAR_REGENERACION:
            print(f"  - Las imágenes ecualizadas ya existen en {dir_salida}. Saltando.")
            continue
            
        # Obtener la lista de imágenes a procesar
        imagenes_a_procesar = list(dir_fuente.glob('*.png'))
        
        if not imagenes_a_procesar:
            print("  - No se encontraron imágenes .png en el directorio fuente.")
            continue

        # Bucle para procesar cada imagen
        for img_path in tqdm(imagenes_a_procesar, desc=f"Ecualizando '{estrategia}'"):
            ecualizar_y_guardar(img_path, dir_salida)

    print("\n--- PASO 3 Completado ---")

if __name__ == "__main__":
    main()