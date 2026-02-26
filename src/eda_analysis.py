import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configuración de estilo
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

FILE_PATH = "dengue_virus_sequences.csv"

def run_eda():
    print(f"📊 Cargando datos desde {FILE_PATH}...")
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo. Asegúrate de ejecutar primero data_extraction.py")
        return

    # 1. Visión General
    print("\n" + "="*40)
    print("RESUMEN GENERAL")
    print("="*40)
    print(f"Total de registros: {len(df)}")
    print(f"Columnas: {list(df.columns)}")
    
    # 2. Análisis de 5'UTR (Objetivo Principal)
    print("\n" + "="*40)
    print("ANÁLISIS DE 5'UTR")
    print("="*40)
    
    # Filtrar filas que NO sean nulas y NO sean vacías
    df_utr = df[df['5UTR'].notna() & (df['5UTR'] != "")].copy()
    
    # --- FILTRADO DE DUPLICADOS ---
    initial_count = len(df_utr)
    df_utr = df_utr.drop_duplicates(subset=['5UTR'])
    unique_count = len(df_utr)
    duplicates_removed = initial_count - unique_count
    
    count_utr = unique_count # Actualizamos para cálculos posteriores
    count_total = len(df)
    percent_utr = (initial_count / count_total) * 100 # Porcentaje original (tenían UTR)
    
    print(f"✅ Secuencias CON información de 5'UTR: {initial_count}")
    print(f"❌ Secuencias SIN información de 5'UTR: {count_total - initial_count}")
    print(f"♻️ Duplicados eliminados (misma secuencia 5'UTR): {duplicates_removed}")
    print(f"📉 Secuencias ÚNICAS finales para análisis: {unique_count}")
    print(f"📊 Porcentaje de utilidad (Total Registros -> Únicos con UTR): {(unique_count / count_total) * 100:.2f}%")
    
    if count_utr == 0:
        print("⚠️ No hay datos con 5'UTR para analizar.")
        return

    # Añadir columna de longitud de 5'UTR
    df_utr['5UTR_length'] = df_utr['5UTR'].apply(len)
    
    print("\nEstadísticas de Longitud de 5'UTR:")
    print(df_utr['5UTR_length'].describe())
    
    # Top 5 5'UTRs más comunes (para ver si hay repetidos/artefactos)
    print("\nEjemplos de 5'UTR más frecuentes:")
    print(df_utr['5UTR'].value_counts().head(5))

    # 3. Visualización
    print("\n🎨 Generando gráficos...")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_utr['5UTR_length'], bins=30, kde=True, color="teal")
    plt.title("Distribución de Longitudes de 5'UTR")
    plt.xlabel("Longitud (nucleótidos)")
    plt.ylabel("Frecuencia")
    plt.savefig("5utr_length_distribution.png")
    print("✔️ Gráfico guardado: 5utr_length_distribution.png")
    
    # Gráfico de pastel: Completitud
    plt.figure(figsize=(6, 6))
    plt.pie([count_utr, count_total - count_utr], labels=['Con 5\'UTR', 'Sin 5\'UTR'], autopct='%1.1f%%', colors=['#4CAF50', '#FF5722'])
    plt.title("Proporción de Datos con 5'UTR")
    plt.savefig("5utr_availability.png")
    print("✔️ Gráfico guardado: 5utr_availability.png")

    # 4. Guardar dataset limpio filtrado
    output_clean = "dengue_5utr_filtrado.csv"
    df_utr.to_csv(output_clean, index=False)
    print(f"\n📂 Archivo filtrado guardado: {output_clean} ({len(df_utr)} registros)")
    print("   -> Este archivo contiene SÓLO las filas que tienen datos en 5'UTR.")

if __name__ == "__main__":
    run_eda()
