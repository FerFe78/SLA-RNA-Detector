# 2_crear_imagenes.py
import torch
from PIL import Image
from tqdm import tqdm
import o_config as config

def main():
    print("--- PASO 2: Creación de Imágenes desde Embeddings ---")
    
    # cargamos el archivo de embeddings variables.
    embeddings_file = config.OUTPUT_DIR / "variable_embeddings.pt"
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Archivo de embeddings no encontrado: {embeddings_file}\nEjecuta el script de generación de embeddings variables primero.")

    image_base_dir = config.IMAGE_DIRS["global"]
    if image_base_dir.exists() and any(image_base_dir.iterdir()) and not config.FORZAR_REGENERACION:
        print("Las carpetas de imágenes ya contienen datos. Saltando este paso.")
        return

    # Crear directorios de salida sin el sufijo de longitud
    for strategy in config.IMAGE_DIRS:
        config.IMAGE_DIRS[strategy].mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando embeddings desde {embeddings_file}...")
    data = torch.load(embeddings_file, map_location=config.DEVICE)
    
    # 'embeddings' ahora es una LISTA de tensores, no un único tensor.
    embeddings_list, labels = data['embeddings'], data['labels']

    # --- Cálculo de Min/Max Global ---
    # Para la normalización global, debemos encontrar el min/max a través de TODOS los tensores en la lista.
    # Concatenamos todos los tensores en uno plano para encontrar los verdaderos valores globales.
    print("Calculando min/max global a través de todos los embeddings...")
    # Usamos un generador para ser eficientes en memoria antes de concatenar
    all_embeddings_tensor = torch.cat([emb.flatten() for emb in embeddings_list])
    global_min, global_max = all_embeddings_tensor.min(), all_embeddings_tensor.max()
    print(f"Min/Max Global: {global_min:.4f}, {global_max:.4f}")

    print("Generando representaciones visuales...")
    # --- Bucle de procesamiento ---
    # Iteramos sobre la lista de embeddings y las etiquetas simultáneamente.
    # Usamos `enumerate` para obtener el índice `i` para el nombre del archivo.
    for i, (emb, label) in enumerate(tqdm(zip(embeddings_list, labels), total=len(labels), desc="Procesando imágenes")):
        # `emb` es ahora un tensor individual (ej. de forma [L, 640])
        # `label` es la etiqueta correspondiente

        # (a) Global: Normaliza usando los valores globales calculados previamente
        if global_max > global_min:
            img_array = ((emb - global_min) / (global_max - global_min) * 255.0).cpu().byte().numpy()
            # --- Ruta de guardado ---
            Image.fromarray(img_array, 'L').save(config.IMAGE_DIRS["global"] / f"{i}-{label}.png")

        # (b) Local: Opera sobre un solo embedding
        local_min, local_max = emb.min(), emb.max()
        if local_max > local_min:
            img_array = ((emb - local_min) / (local_max - local_min) * 255.0).cpu().byte().numpy()
            Image.fromarray(img_array, 'L').save(config.IMAGE_DIRS["local"] / f"{i}-{label}.png")


    print("--- PASO 2 Completado ---")

if __name__ == "__main__":
    main()