# 1_generar_embeddings.py
import pandas as pd
import torch
import fm
from tqdm import tqdm
import numpy as np
import argparse

import o_config as config

torch.serialization.add_safe_globals([argparse.Namespace])

def main():
    print("--- PASO 1: Generación de Embeddings de Longitud Variable ---")
    
    # El nombre del archivo ahora refleja que los embeddings son de longitud variable.
    output_file = config.OUTPUT_DIR / "variable_embeddings.pt"
    if output_file.exists() and not config.FORZAR_REGENERACION:
        print(f"El archivo de embeddings ya existe en: {output_file}. Saltando este paso.")
        return

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando modelo RNA-FM desde: {config.RNAFM_MODEL_PATH}")
    model, alphabet = fm.pretrained.rna_fm_t12(str(config.RNAFM_MODEL_PATH))
    model.to(config.DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()

    df = pd.read_csv(config.CSV_PATH, sep=';', header=0, encoding='latin1', skipinitialspace=True)
    df.columns = ['nombre', 'secuencia', 'sla']
    
    print(f"Procesando {len(df)} secuencias con sus longitudes originales...")
    data = [(row['nombre'], row['secuencia']) for _, row in df.iterrows()]
    labels = [int(row['sla']) for _, row in df.iterrows()]
    
    lista_embeddings = []
    with torch.no_grad():
        for nombre, secuencia in tqdm(data, total=len(data), desc="Generando Embeddings (uno a uno)"):
            # Obtenemos la longitud real de la secuencia actual
            longitud_real = len(secuencia)
            
            data_single = [(nombre, secuencia)]
            _, _, tokens = batch_converter(data_single)
            out = model(tokens.to(config.DEVICE), repr_layers=[12])
            # Extraemos el embedding de la capa 12, que es de longitud variable
            # y puede ser de longitud diferente para cada secuencia.
            emb = out['representations'][12][0, 1:longitud_real + 1, :]
            
            lista_embeddings.append(emb.cpu())
            
    # Convertimos la lista de tensores a un tensor de PyTorch.
    print(f"Guardando lista de {len(lista_embeddings)} tensores de embeddings en: {output_file}")
    # El objeto guardado sigue siendo un diccionario, pero 'embeddings' es ahora una lista de tensores.
    torch.save({'embeddings': lista_embeddings, 'labels': labels}, output_file)
    print("--- PASO 1 Completado ---")

if __name__ == "__main__":
    main()