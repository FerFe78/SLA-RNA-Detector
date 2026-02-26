import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

try:
    import o_config as config
    ruta_archivo = config.CSV_PATH
except ImportError:
    ruta_archivo = "base_de_datos_SLA.csv"

# 1. Cargar datos
try:
    df = pd.read_csv(ruta_archivo, sep=';', header=0, encoding='utf-8')
except:
    df = pd.read_csv(ruta_archivo, sep=';', header=0, encoding='latin1')

# Normalizar nombres de columnas a minúsculas
df.columns = [c.lower() for c in df.columns]

# 2. Estadísticas Básicas
conteo_sla = df['sla'].value_counts()
print("--- Resumen de SLA ---")
print(conteo_sla)
print(f"\nInfo del DataFrame:")
df.info()

# 3. Análisis de Duplicados (Eficiente)
print("\n--- Análisis de Duplicados ---")
n_unicos_sec = df['secuencia'].nunique()
n_unicos_nom = df['nombre'].nunique()
print(f"Secuencias únicas: {n_unicos_sec} / {len(df)}")
print(f"Nombres únicos: {n_unicos_nom} / {len(df)}")

# Encontrar duplicados específicos si existen
dup_sec = df[df.duplicated('secuencia', keep=False)]
if not dup_sec.empty:
    print(f"Se encontraron {len(dup_sec)} filas con secuencias duplicadas.")

# 4. Distribución de Largos (Vectorizado)
df['largo'] = df['secuencia'].str.len()
stats_largos = df.groupby('sla')['largo'].describe()
print("\n--- Estadísticas de Largo por Clase ---")
print(stats_largos)

# 5. Conteo de Bases (Vectorizado y Rápido)
# Unimos todas las secuencias y usamos Counter
print("\n--- Composición de Bases ---")
todas_las_bases = "".join(df['secuencia'].astype(str))
conteo_bases = Counter(todas_las_bases)
print("Distribución de nucleótidos:", dict(conteo_bases))

# Verificar caracteres válidos
caracteres_esperados = {'A', 'G', 'C', 'U', 'N'}
caracteres_extra = set(conteo_bases.keys()) - caracteres_esperados
if caracteres_extra:
    print(f"⚠️ ¡Atención! Se detectaron caracteres inesperados: {caracteres_extra}")

print(f"\nTotal de bases procesadas: {sum(conteo_bases.values())}")
print(f"Secuencias Positivas: {len(df[df['sla']==1])}")
print(f"Secuencias Negativas: {len(df[df['sla']==0])}")
