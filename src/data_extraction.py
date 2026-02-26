from Bio import Entrez, SeqIO
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= CONFIGURACIÓN =================
Entrez.email = "lopezortsfernando@gmail.com"
Entrez.api_key = None  # Si tienes una API KEY de NCBI, ponla aquí para aumentar la velocidad (permite 10 req/s frente a 3 req/s)
TAXID = "12637"  # Dengue virus
OUTPUT_FILE = "dengue_virus_sequences.csv"
RETMAX = 100000
BATCH_SIZE = 50 
MAX_WORKERS = 10 if Entrez.api_key else 3 # Paralelismo seguro (3 sin API key, 10 con API key)
# =================================================

def process_batch(batch_ids, batch_index, total_batches):
    """Descarga y procesa un lote de IDs en un hilo separado."""
    rows = []
    try:
        print(f"⬇️ Iniciando lote {batch_index}/{total_batches} ({len(batch_ids)} seqs)...")
        handle = Entrez.efetch(
            db="nucleotide",
            id=batch_ids,
            rettype="gb",
            retmode="text"
        )
        records = SeqIO.parse(handle, "genbank")
        
        for record in records:
            try:
                accession = record.id
                organism = record.annotations.get("organism", "")
                species = " ".join(organism.split()[:2]) if organism else "Unknown"
                length = len(record.seq)

                definition = record.description.lower()
                nuc_completeness = "complete" if "complete genome" in definition else "partial"

                host = ""
                for feature in record.features:
                    if feature.type == "source" and "host" in feature.qualifiers:
                        host = feature.qualifiers["host"][0]
                        break
                
                # --- FILTRO: Excluir Homo sapiens ---
                if "homo sapiens" not in host.lower():
                    continue

                sequence = str(record.seq)

                # ---- 5'UTR ----
                five_utr = ""
                cds_starts = [int(f.location.start) for f in record.features if f.type == "CDS"]
                if cds_starts:
                    min_start = min(cds_starts)
                    if min_start > 0:
                        five_utr = sequence[:min_start]

                rows.append([
                    accession, organism, species, length, 
                    nuc_completeness, host, sequence, five_utr
                ])
                
            except Exception as rec_e:
                print(f"⚠️ Error procesando registro en lote {batch_index}: {rec_e}")
                continue
        
        handle.close()
        return rows

    except Exception as batch_e:
        print(f"❌ Error grave descargando lote {batch_index}: {batch_e}")
        return []

def fetch_sequences():
    print("🔍 Buscando secuencias del virus del dengue en NCBI...")
    
    try:
        search_handle = Entrez.esearch(
            db="nucleotide",
            term=f"txid{TAXID}[Organism:exp]",
            retmax=RETMAX,
            usehistory="y"
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
    except Exception as e:
        print(f"❌ Error en la búsqueda: {e}")
        return

    id_list = search_results["IdList"]
    count = int(search_results["Count"])
    
    print(f"✔️ Se encontraron {count} secuencias disponibles.")
    print(f"✔️ Se descargarán hasta {len(id_list)} secuencias.")
    print(f"🚀 Ejecución en PARALELO con {MAX_WORKERS} hilos simultáneos.")

    if not id_list:
        print("⚠️ No se encontraron secuencias.")
        return

    # Preparar lotes
    batches = [id_list[i:i + BATCH_SIZE] for i in range(0, len(id_list), BATCH_SIZE)]
    total_batches = len(batches)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["accession", "organism name", "species", "length", "nuc completeness", "host", "secuencia", "5UTR"])

        total_downloaded = 0
        
        # Ejecutor de Hilos
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Enviar tareas
            future_to_batch = {
                executor.submit(process_batch, batch, i+1, total_batches): i 
                for i, batch in enumerate(batches)
            }
            
            # Recoger resultados conforme terminan
            for future in as_completed(future_to_batch):
                batch_rows = future.result()
                if batch_rows:
                    writer.writerows(batch_rows)
                    total_downloaded += len(batch_rows)
                    if total_downloaded % 500 == 0:
                        print(f"📈 Progreso: {total_downloaded} secuencias guardadas...")

    print(f"\n✅ Proceso completado.")
    print(f"📂 Archivo generado: {OUTPUT_FILE}")
    print(f"📊 Total secuencias guardadas: {total_downloaded}")

if __name__ == "__main__":
    fetch_sequences()
