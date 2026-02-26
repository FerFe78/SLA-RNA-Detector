import pandas as pd
import os

INPUT_FILE = "code/data/dengue_5utr_filtrado.csv"
OUTPUT_DIR = "code/data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dengue_training_full.csv")

def convert_data():
    print(f"🔄 Leyendo {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo de entrada.")
        return

    # Selección y Renombrado de columnas
    # Formato esperado: 'nombre', 'secuencia', 'sla'
    # nombre <- accession
    # secuencia <- 5UTR
    # sla <- 1 (Asumimos que todas estas 5'UTRs contienen estructuras SLA o son de interés positivo)
    
    print("🛠️ Transformando datos...")
    df_train = pd.DataFrame()
    df_train['nombre'] = df["accession"]
    df_train['secuencia'] = df["5UTR"]
    df_train['sla'] = 1 # Etiqueta Positiva
    
    # Asegurar directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Guardar con separador ';' como espera el script de entrenamiento
    print(f"💾 Guardando dataset de entrenamiento en {OUTPUT_FILE}...")
    df_train.to_csv(OUTPUT_FILE, sep=';', index=False, encoding='latin1') # Script 1 usa encoding latin1
    
    print(f"✅ Conversión completada. {len(df_train)} registros listos para entrenar.")
    print(f"   Columnas: {list(df_train.columns)}")

if __name__ == "__main__":
    convert_data()
