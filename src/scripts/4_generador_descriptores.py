# 4_generar_descriptores.py

import cv2
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import o_config as config

def main():
    print("--- PASO 4: Generación de Descriptores de Características (Versión Adaptada) ---")

    # --- 1. Inicializar detectores y descriptores disponibles ---
    # Usamos algoritmos libres de patentes o cuya patente ha expirado y están incluidos.
    try:
        detectors = {
            'sift': cv2.SIFT_create(),
            'brisk': cv2.BRISK_create(),
            'fast': cv2.FastFeatureDetector_create(),
            'orb': cv2.ORB_create(nfeatures=2000) # ORB es un detector y descriptor a la vez
        }
        
        descriptors = {
            'sift': cv2.SIFT_create(),
            'brisk': cv2.BRISK_create(),
            'brief': cv2.xfeatures2d.BriefDescriptorExtractor_create(),
            'orb': cv2.ORB_create(nfeatures=2000)
        }
    except AttributeError as e:
        print(f"\nERROR: Faltan componentes de OpenCV ({e}).")
        print("Asegúrate de tener 'opencv-contrib-python' instalado.")
        print("Ejecuta: pip install opencv-contrib-python")
        return

    # --- 2. Definir las combinaciones a probar ---
    combinations_to_test = [
        ('sift', 'sift'),       # SIFT completo
        ('brisk', 'brisk'),     # BRISK completo
        ('orb', 'orb'),         # NUEVO: ORB completo
        ('fast', 'brisk'),      # FAST + BRISK
        ('fast', 'brief'),      # FAST + BRIEF
        ('fast', 'sift')        # FAST + SIFT
    ]

    # Estrategias de imagen a procesar
    estrategias_fuente_eq = ["global_eq", "local_eq"]
    config.DESCRIPTORS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 3. Bucle principal refactorizado ---
    for det_name, desc_name in combinations_to_test:
        combination_name = f"{det_name}_{desc_name}"
        print(f"\n===== Procesando combinación: {combination_name.upper()} =====")
        
        detector_obj = detectors[det_name]
        descriptor_obj = descriptors[desc_name]
        
        for estrategia in estrategias_fuente_eq:
            output_file = config.DESCRIPTORS_DIR / f"{combination_name}_{estrategia}.pkl"
            dir_fuente = config.IMAGE_DIRS[estrategia]
            
            print(f"  - Estrategia: '{estrategia}' -> Archivo de salida: {output_file.name}")

            if output_file.exists() and not config.FORZAR_REGENERACION:
                print("    - El archivo de descriptores ya existe. Saltando.")
                continue

            if not dir_fuente.exists() or not any(dir_fuente.iterdir()):
                print(f"    - ADVERTENCIA: El directorio fuente '{dir_fuente}' está vacío. Saltando.")
                continue

            imagenes_a_procesar = sorted(list(dir_fuente.glob('*.png')))
            
            all_descriptors_list, all_labels, all_image_names = [], [], []

            for img_path in tqdm(imagenes_a_procesar, desc=f"  - Extrayendo {combination_name.upper()}"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                label = int(img_path.stem.split('-')[1])
                
                # SIFT y BRISK pueden hacer detectAndCompute directamente
                if det_name == desc_name and det_name in ['sift', 'brisk', 'orb']:
                    keypoints, des = detector_obj.detectAndCompute(img, None)
                else:
                    # Lógica general para combinaciones
                    keypoints = detector_obj.detect(img, None)
                    keypoints, des = descriptor_obj.compute(img, keypoints)

                if des is not None and len(des) > 0:
                    all_descriptors_list.append(des)
                    all_labels.append(label)
                    all_image_names.append(img_path.name)
            
            if all_descriptors_list:
                data_to_save = {
                    'features': all_descriptors_list, 'labels': all_labels, 'image_names': all_image_names
                }
                with open(output_file, 'wb') as f:
                    pickle.dump(data_to_save, f)
                print(f"    - ¡Éxito! Se guardaron los descriptores de {len(all_descriptors_list)} imágenes.")
            else:
                print("    - ADVERTENCIA: No se extrajo ningún descriptor para esta combinación.")

    print("\n--- PASO 4 Completado ---")

if __name__ == "__main__":
    main()