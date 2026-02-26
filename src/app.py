import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
import sys
import os

# Asegurar que el directorio 'src' esté en el path para encontrar sla_detector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sla_detector import config, embeddings, image_processing, features, classifier
import ui_utils  # Import custom UI helper

# --- Configuración Inicial ---
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Detector de Stem-Loop A",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar estilos personalizados
ui_utils.load_css()

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.markdown("## 🧬 Configuración")
    st.markdown("Ajusta los parámetros del detector y el modelo.")
    st.divider()

    # 1. Selector de Modo
    st.markdown("### 📍 Modo de Operación")
    modo = st.radio(
        "Seleccione tarea:",
        ["Analizar Fragmento(s)", "Escanear Genoma Completo"],
        captions=["Secuencias cortas (60-90nt)", "Ventana deslizante sobre genoma"]
    )
    
    st.divider()
    
    # 2. Selector de Versión del Modelo
    st.markdown("### 🤖 Modelo IA")
    version_modelo = st.selectbox(
        "Conjunto de Entrenamiento", 
        ["Original", "Mixto (Reentrenado)"],
        help="Original: Datos reales (Dengue, Zika). Mixto: Incluye datos sintéticos."
    )

    # Definir directorio según versión
    if version_modelo == "Original":
        target_dir = config.INTERMEDIATE_MODELS_DIR
    else:
        target_dir = config.INTERMEDIATE_MODELS_MIXED_DIR

    # Listar modelos disponibles
    try:
        if not target_dir.exists():
            st.warning(f"Directorio {version_modelo} no encontrado.")
            all_model_files = []
        else:
            all_model_files = [p.name for p in target_dir.glob('*.pkl')]
    except Exception as e:
        st.error(f"Error filesystem: {e}")
        st.stop()
        
    local_model_files = [f for f in all_model_files if '_local_' in f]

    if not local_model_files:
        st.warning("⚠️ No hay modelos disponibles.")
        model_file = None
    else:
        model_file = st.selectbox(
            "Arquitectura Específica",
            local_model_files,
            index=0,
            format_func=lambda x: x.replace(".pkl", "").replace("pipeline_", "")
        )

    if model_file:
        model_path = target_dir / model_file
        model_params = classifier.parse_model_filename(model_file)
        if model_params:
            st.caption(f"Detector: **{model_params['detector']}** | Descriptor: **{model_params['descriptor']}**")
    else:
        st.stop()

# --- Carga de Modelos (Backend) ---
# Se mantiene igual, pero validamos que todo cargue ok antes de pintar la UI principal
if model_file:
    rna_fm_model, rna_fm_alphabet = embeddings.load_rna_fm_model()
    try:
        clasificador_components = classifier.load_classifier(model_path)
        kmeans_model = clasificador_components.get('kmeans_vocab')
        classifier_model = clasificador_components.get('classifier') 
        k_value = clasificador_components.get('config', {}).get('vocab_size', 500)
    except Exception as e:
        st.error(f"Error crítico cargando clasificador: {e}")
        st.stop()

# --- Lógica de Predicción ---
def predict_single(sequence, detector, descriptor, kmeans, clf, k):
    """Pipeline completo para una sola secuencia."""
    emb = embeddings.get_single_embedding(sequence)
    img_eq, _ = image_processing.process_embedding_to_image(emb)
    desc, _ = features.extract_features(img_eq, detector, descriptor)
    hist = features.create_bow_histogram(kmeans, desc, k)
    pred = clf.predict(hist)[0]
    prob = clf.predict_proba(hist)[0] if hasattr(clf, "predict_proba") else [0.0, 0.0]
    return pred, prob, emb, img_eq


# --- Interfaz Principal ---

if modo == "Analizar Fragmento(s)":
    ui_utils.display_header("Analizar Fragmento(s)", "Evalúa secuencias cortas de ARN para detectar estructuras SLA.")

    tab1, tab2 = st.tabs(["⚡ Análisis Individual", "📂 Carga por Lote (CSV/FASTA)"])
    
    with tab1:
        col_input, col_res = st.columns([1, 1], gap="large")
        
        with col_input:
            st.write("#### Entrada")
            if "seq_input_area" not in st.session_state:
                st.session_state.seq_input_area = ""

            def clear_indiv():
                st.session_state.seq_input_area = ""
                
            secuencia_input = st.text_area(
                "Pegar secuencia de ARN:", 
                height=200, 
                key="seq_input_area",
                placeholder="Ej: GGG...CCC"
            )
            
            c1, c2 = st.columns([1, 3])
            with c1:
                if st.button("Limpiar", on_click=clear_indiv):
                    pass
            with c2:
                analyze_btn = st.button("Analizar Secuencia", type="primary", width="stretch")

        with col_res:
            st.write("#### Resultados")
            if analyze_btn and secuencia_input:
                if len(secuencia_input) < 10:
                    st.warning("La secuencia parece muy corta.")
                else:
                    with st.spinner("Analizando estructura..."):
                        pred, prob, _, img_eq = predict_single(
                            secuencia_input, model_params['detector'], model_params['descriptor'],
                            kmeans_model, classifier_model, k_value
                        )
                        ui_utils.display_result_card(pred, prob[1], img_eq)
            elif analyze_btn and not secuencia_input:
                st.info("👈 Por favor ingresa una secuencia para comenzar.")
            else:
                st.write("Waiting for input...")

    with tab2:
        st.info("Soportamos CSV (Columnas: 'id', 'secuencia') o FASTA estándar.")
        uploaded_file = st.file_uploader("Arrastra tu archivo aquí", type=["csv", "fasta", "fa", "txt"])
        
        if uploaded_file:
            # Reutilizamos lógica de carga de DF existente pero limpiamos la presentación
            df = None
            filename = uploaded_file.name.lower()
            
            try:
                if filename.endswith('.csv'):
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    # Intentar fix de separador ;
                    if len(df.columns) < 2:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
                    
                    df.columns = [c.lower().strip() for c in df.columns]
                    # Renombrar columnas
                    rename_map = {'nombre': 'id', 'accession': 'id', 'acc': 'id', 
                                  'frequency': 'secuencia', 'sequence': 'secuencia', 'seq': 'secuencia'}
                    df = df.rename(columns=rename_map)
                    
                    if 'id' not in df.columns or 'secuencia' not in df.columns:
                        st.error("El CSV debe tener columnas 'id' y 'secuencia'.")
                        df = None

                else: # FASTA
                    from io import StringIO
                    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                    fasta_data = []
                    header = None
                    seq_parts = []
                    for line in stringio:
                        line = line.strip()
                        if not line: continue
                        if line.startswith(">"):
                            if header: fasta_data.append({"id": header, "secuencia": "".join(seq_parts)})
                            header = line[1:].split()[0]
                            seq_parts = []
                        else:
                            seq_parts.append(line)
                    if header: fasta_data.append({"id": header, "secuencia": "".join(seq_parts)})
                    if fasta_data: df = pd.DataFrame(fasta_data)
            
            except Exception as ex:
                st.error(f"Error procesando archivo: {ex}")

            if df is not None:
                st.dataframe(df.head(), width="stretch")
                
                if st.button(f"Procesar {len(df)} secuencias", type="primary"):
                    sequences = [(str(row['id']), str(row['secuencia'])) for _, row in df.iterrows()]
                    results_data = []
                    prog_bar = st.progress(0)
                    
                    # Batch Embeddings
                    with st.spinner("Generando representaciones..."):
                        emb_dict = embeddings.get_batch_embeddings(sequences, batch_size=8)
                    
                    total = len(sequences)
                    for i, (name, seq) in enumerate(sequences):
                        if name in emb_dict:
                            emb = emb_dict[name]
                            img_eq, _ = image_processing.process_embedding_to_image(emb)
                            desc, _ = features.extract_features(img_eq, model_params['detector'], model_params['descriptor'])
                            hist = features.create_bow_histogram(kmeans_model, desc, k_value)
                            
                            pred = classifier_model.predict(hist)[0]
                            prob = classifier_model.predict_proba(hist)[0][1] if hasattr(classifier_model, "predict_proba") else 0
                            
                            results_data.append({"id": name, "prediccion": "SLA" if pred==1 else "NO SLA", "probabilidad_sla": prob})
                        prog_bar.progress((i+1)/total)
                        
                    res_df = pd.DataFrame(results_data)
                    st.success("✅ Procesamiento completado")
                    st.dataframe(res_df, width="stretch")
                    
                    csv = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Descargar CSV", csv, "resultados_sla.csv", "text/csv")


elif modo == "Escanear Genoma Completo":
    ui_utils.display_header("Escáner Genómico", "Detecta regiones SLA en secuencias largas usando ventanas deslizantes.")
    
    # Device Info
    device_status = "🟢 GPU Activada (CUDA)" if config.DEVICE == 'cuda' else "🟡 CPU (Lento)"
    st.caption(f"Dispositivo de procesamiento: **{device_status}**")
    
    with st.container():
        c1, c2 = st.columns([2, 1])
        seq_full = None
        
        with c1:
            method = st.radio("Entrada:", ["Cargar Archivo", "Pegar Texto"], horizontal=True, label_visibility="collapsed")
            if method == "Cargar Archivo":
                gf = st.file_uploader("Archivo Genoma (FASTA/TXT)", type=["fasta", "txt", "fa"])
                if gf:
                    c = gf.read().decode("utf-8")
                    seq_full = "".join(c.split("\n", 1)[1].splitlines()) if c.startswith(">") else "".join(c.splitlines())
            else:
                if "genome_input_area" not in st.session_state: st.session_state.genome_input_area = ""
                
                def clear_genome_input():
                    st.session_state.genome_input_area = ""
                    st.session_state.scan_results = None

                gt = st.text_area("Secuencia completa:", height=150, key="genome_input_area")
                st.button("Limpiar texto", on_click=clear_genome_input)
                
                if gt:
                     seq_full = "".join(gt.strip().split("\n", 1)[1].splitlines()) if gt.strip().startswith(">") else "".join(gt.splitlines())
            
            # Display Sequence Length
            if seq_full:
                st.info(f"📏 Longitud de la secuencia: **{len(seq_full)}** nucleótidos")

        with c2:
            st.write("**Configuración de Ventana**")
            
            multi_scale = st.checkbox("📶 Modo Multi-Escala", help="Analiza múltiples tamaños de ventana para detectar SLA de longitud variable.")
            
            # New Toggle for Heatmap
            show_heatmap = st.checkbox("🔥 Generar Mapa de Calor", value=True, help="Desactiva para mejorar rendimiento en escaneos muy grandes.")
            
            if multi_scale:
                w_min = st.number_input("Min Tamaño (nt)", 50, 200, 60)
                w_max = st.number_input("Max Tamaño (nt)", 50, 200, 100)
                w_step = st.number_input("Paso (nt)", 1, 50, 5)
                
                window_sizes = range(w_min, w_max + 1, w_step)
                st.caption(f"Analizará {len(window_sizes)} tamaños: {list(window_sizes)}")
                s_size = st.number_input("Desplazamiento (Stride)", 1, 100, 10)
            else:
                # Single Scale
                window_sizes = [st.slider("Tamaño Ventana", 60, 100, 80)]
                s_size = st.slider("Paso (Step)", 5, 50, 20)
            
            if st.button("🚀 Iniciar Escaneo", type="primary", width="stretch", disabled=(not seq_full)):
                if not seq_full: st.stop()
                st.session_state.scan_results = None
                st.session_state.is_multi_scale = multi_scale
                st.session_state.show_heatmap_pref = show_heatmap # Store preference
                
                seq_cleaned = seq_full.upper().replace("\n", "").replace(" ", "").strip()
                
                # --- GLOBAL BATCHING GENERATION ---
                all_windows = []
                # Metadata list to track original info for each window
                # List of tuples: (pos, size, seq)
                meta_data = [] 
                
                start_time = time.time() # Start Timer
                
                for w_size in window_sizes:
                    if len(seq_cleaned) < w_size: continue
                    
                    for i in range(0, len(seq_cleaned) - w_size + 1, s_size):
                        w_seq = seq_cleaned[i : i + w_size]
                        # ID format: "pos_SIZE" (Collision if not unique, so use index) -> actually get_batch_embeddings uses list, dict keys must be unique.
                        # We will use a unique ID: f"{w_size}_{i}"
                        uid = f"{w_size}_{i}"
                        all_windows.append((uid, w_seq))
                        meta_data.append({"Posición": i, "Tamaño Ventana": w_size, "Secuencia": w_seq, "ID": uid})
                
                if not all_windows:
                    st.error("Secuencia demasiado corta.")
                else:
                    final_data = []
                    status_placeholder = st.empty()
                    prog_bar = st.progress(0)
                    
                    total_wins = len(all_windows)
                    status_placeholder.info(f"🚀 Iniciando escaneo de {total_wins} ventanas en {len(window_sizes)} escalas...")
                    
                    # Optimized UI batching
                    chunk_size = 128 if config.DEVICE == 'cuda' else 32
                    
                    start_time = time.time()
                    for i in range(0, total_wins, chunk_size):
                        chunk = all_windows[i : i + chunk_size]
                        
                        # Use a sub-spinner for cleaner UI
                        status_placeholder.write(f"🧬 Procesando lote {i//chunk_size + 1} ({i}-{min(i+chunk_size, total_wins)} de {total_wins})")
                        
                        # Batch Inference
                        emb_dict = embeddings.get_batch_embeddings(chunk, batch_size=16)
                        
                        # Process individual results in the chunk
                        for j, (uid, _) in enumerate(chunk):
                            if uid in emb_dict:
                                meta = meta_data[i + j]
                                emb = emb_dict[uid]
                                
                                # Visual process
                                img_eq, _ = image_processing.process_embedding_to_image(emb)
                                desc, _ = features.extract_features(img_eq, model_params['detector'], model_params['descriptor'])
                                hist = features.create_bow_histogram(kmeans_model, desc, k_value)
                                prob = classifier_model.predict_proba(hist)[0][1] if hasattr(classifier_model, "predict_proba") else 0
                                
                                final_data.append({
                                    "Posición": meta["Posición"],
                                    "Tamaño Ventana": meta["Tamaño Ventana"],
                                    "Probabilidad SLA": prob,
                                    "Secuencia": meta["Secuencia"]
                                })

                        pct = (i + chunk_size) / total_wins
                        prog_bar.progress(min(1.0, pct))
                    
                    status_placeholder.success(f"✅ Escaneo completado con éxito en {time.time() - start_time:.2f}s")
                    st.session_state.scan_results = pd.DataFrame(final_data)
                    st.session_state.process_time = time.time() - start_time

    # Display Scan Results
    if "scan_results" in st.session_state and st.session_state.scan_results is not None:
        st.divider()
        res = st.session_state.scan_results
        
        # 1. Processing Time (Top of Results)
        if "process_time" in st.session_state:
            st.success(f"⏱️ Tiempo de procesamiento: **{st.session_state.process_time:.2f} segundos**")
        
        is_multi = st.session_state.get("is_multi_scale", False)
        show_map = st.session_state.get("show_heatmap_pref", True)
        
        if is_multi:
            if show_map:
                st.subheader("🔥 Mapa de Calor Multi-Escala")
                ui_utils.plot_heatmap(res)
            
            st.subheader("📈 Proyección de Máxima Probabilidad")
            # Group by Position, take Max Prob
            max_proj = res.groupby("Posición")["Probabilidad SLA"].max().reset_index()
            st.line_chart(max_proj, x="Posición", y="Probabilidad SLA", color="#e63946")
        else:
            if show_map:
                 st.subheader("📊 Mapa de Probabilidad")
                 st.line_chart(res, x="Posición", y="Probabilidad SLA", color="#e63946")
        
        c_top, c_ver = st.columns([1, 1], gap="large")
        
        with c_top:
            st.write("### 🏆 Top 5 Candidatos (Global)")
            top_5 = res.sort_values(by="Probabilidad SLA", ascending=False).head(5).reset_index(drop=True)
            
            # Table View
            cols_to_show = ["Posición", "Tamaño Ventana", "Probabilidad SLA"] if "Tamaño Ventana" in top_5.columns else ["Posición", "Probabilidad SLA"]
            st.dataframe(
                top_5[cols_to_show].style.format({"Probabilidad SLA": "{:.2%}"}), 
                width="stretch",
                height=200
            )

            # Detail Inspector
            st.write("#### 👁️ Detalle y Copia")
            
            # Format: #{i} | Pos: Start-End (L=Len) | Prob: %
            def fmt_cand(i):
                row = top_5.loc[i]
                start = row['Posición']
                # Determine length (either from column or len(seq))
                length = row['Tamaño Ventana'] if 'Tamaño Ventana' in row else len(row['Secuencia'])
                end = start + length
                return f"#{i+1} | Pos: {start}-{end} (L={length}) | Prob: {row['Probabilidad SLA']:.1%}"

            cand_idx = st.selectbox(
                "Selecciona un candidato para ver su secuencia:",
                top_5.index,
                format_func=fmt_cand
            )
            
            if cand_idx is not None:
                sel_row = top_5.loc[cand_idx]
                st.code(sel_row['Secuencia'], language="text")
                win_len = sel_row.get('Tamaño Ventana', len(sel_row['Secuencia']))
                st.caption(f"Posición: {sel_row['Posición']} - {sel_row['Posición'] + win_len} (Long: {win_len})")

        with c_ver:
            st.write("### 🔎 Verificación Cruzada")
            st.info("Selecciona otro modelo para validar los candidatos.")
            
            verify_model = st.selectbox("Modelo Verificador", local_model_files, key="v_mod_sel")
            
            if st.button("Verificar Candidatos"):
                # Load verify model
                v_path = target_dir / verify_model
                v_comps = classifier.load_classifier(v_path)
                v_k = v_comps.get('config', {}).get('vocab_size', 500)
                v_mod_p = classifier.parse_model_filename(verify_model)
                
                ver_res = []
                for idx, row in top_5.iterrows():
                    seq = row['Secuencia']
                    # Verify predict
                    # (Simplified call for brevity, in real code ensure same pipeline reused)
                    emb = embeddings.get_single_embedding(seq) # Can reuse if cached? or just recalc for 5 items is fast
                    img_eq, _ = image_processing.process_embedding_to_image(emb)
                    desc, _ = features.extract_features(img_eq, v_mod_p['detector'], v_mod_p['descriptor'])
                    hist = features.create_bow_histogram(v_comps['kmeans_vocab'], desc, v_k)
                    prob_v = v_comps['classifier'].predict_proba(hist)[0][1]
                    
                    ver_res.append({
                        "Pos": row['Posición'],
                        "Orig": f"{row['Probabilidad SLA']:.2f}",
                        "Verif": f"{prob_v:.2f}",
                        "Status": "✅" if abs(row['Probabilidad SLA'] - prob_v) < 0.15 else "⚠️"
                    })
                
                st.table(pd.DataFrame(ver_res))

        # 3. Sequence Extractor Tool (Keep at bottom)
        with st.container():
            st.divider()
            st.subheader("✂️ Extraer y Analizar Sub-secuencia")
            with st.expander("Abrir herramienta de extracción manual"):
                e_col1, e_col2, e_col3 = st.columns([1, 1, 1])
                with e_col1:
                    start_pos_in = st.number_input("Desde (Posición)", min_value=0, max_value=len(seq_full) if seq_full else 1000, value=0)
                with e_col2:
                    end_pos_in = st.number_input("Hasta (Posición)", min_value=0, max_value=len(seq_full) if seq_full else 1000, value=min(100, len(seq_full) if seq_full else 100))
                with e_col3:
                    st.write("") # Spacer
                    st.write("") 
                    extract_btn = st.button("Extraer", width="stretch")
                
                if extract_btn and seq_full:
                    if start_pos_in >= end_pos_in:
                        st.error("El inicio debe ser menor al final.")
                    else:
                        sub_seq = seq_full[start_pos_in:end_pos_in]
                        st.info(f"Fragmento extraído ({len(sub_seq)} nt):")
                        st.code(sub_seq, language="text")
                        
                        if st.button("⚡ Analizar este fragmento"):
                             with st.spinner("Analizando..."):
                                # Real call
                                emb_s = embeddings.get_single_embedding(sub_seq)
                                img_s, _ = image_processing.process_embedding_to_image(emb_s)
                                desc_s, _ = features.extract_features(img_s, model_params['detector'], model_params['descriptor'])
                                hist_s = features.create_bow_histogram(kmeans_model, desc_s, k_value)
                                prob_s = classifier_model.predict_proba(hist_s)[0][1]
                                ui_utils.display_result_card(1 if prob_s > 0.5 else 0, prob_s, img_s)
