# limpieza.py
import pandas as pd
import charset_normalizer
import pathlib
import sys

# Try to import config, handle missing file gracefully
try:
    import o_config as config
except ImportError:
    config = None

def detectar_encoding(path: str) -> str:
    """Detects the file encoding using charset_normalizer."""
    with open(path, 'rb') as f:
        matches = charset_normalizer.from_bytes(f.read())
        return matches.best().encoding if matches.best() else 'utf-8'

def limpiar_sla_csv(input_csv: str, output_csv: str):
    """
    Cleans the SLA input CSV: replaces T->U, drops duplicates, maps SLA labels.
    """
    if not pathlib.Path(input_csv).exists():
        print(f"Error: No se encontró el archivo de entrada {input_csv}")
        return

    # 1. Detect encoding and read CSV
    enc = detectar_encoding(input_csv)
    print(f"Detectado encoding: {enc}")
    
    try:
        df = pd.read_csv(
            input_csv,
            sep=';',
            header=0,
            encoding=enc,
            skipinitialspace=True
        )
    except Exception as e:
        print(f"Error leyendo CSV: {e}")
        return

    # Validación básica de columnas
    if len(df.columns) < 3:
        print(f"Advertencia: El CSV tiene menos de 3 columnas. Intentando con separador coma...")
        df = pd.read_csv(input_csv, header=0, encoding=enc)
    
    if len(df.columns) >= 3:
        df.columns = ['nombre', 'secuencia', 'SLA'][:len(df.columns)]
    else:
        print("Error: Estructura de CSV no válida.")
        return

    # 1.1 Reemplazar 'T' por 'U' en las secuencias
    df['secuencia'] = df['secuencia'].astype(str).str.upper().str.replace('T', 'U', regex=False)

    # 2. Eliminar registros idénticos
    df = df.drop_duplicates()

    # 3. Eliminar secuencias duplicadas (distinto nombre, misma secuencia)
    df = df.drop_duplicates(subset=['secuencia'], keep='first')

    # 4. Mapear '+' → 1, '-' → 0 en la columna SLA
    # Handle cases where SLA might already be numeric or different labels
    df['SLA'] = df['SLA'].astype(str).str.strip().map({'+': 1, '-': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0})

    # 5. Guardar CSV limpio (usamos UTF-8 para evitar problemas de compatibilidad futuros)
    df.to_csv(output_csv, sep=';', index=False, encoding='utf-8')
    print(f'Guardado CSV limpio en: {output_csv} (Total: {len(df)} registros)')

if __name__ == '__main__':
    if config:
        entrada = config.RAW_CSV_PATH
        salida  = config.CSV_PATH
    else:
        # Fallback for testing
        entrada = "base_de_datos_SLA.csv"
        salida  = "base_de_datos_SLA_limpia.csv"
        
    limpiar_sla_csv(entrada, salida)