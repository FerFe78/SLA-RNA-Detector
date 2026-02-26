import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_css():
    """Injects a professional 'Stealth Bio-Medical' Dark Mode theme."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
        
        :root {
            --primary-bg: #0f172a;        /* Deep Navy */
            --secondary-bg: #1e293b;      /* Slate */
            --accent: #2dd4bf;            /* Teal */
            --text-main: #f1f5f9;         /* Off-white */
            --text-dim: #94a3b8;          /* Muted Slate */
            --success: #10b981;           /* Emerald */
            --error: #fb7185;             /* Coral/Rose */
            --card-border: rgba(255,255,255,0.1);
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--text-main);
        }

        .stApp {
            background-color: var(--primary-bg);
        }

        /* Sidebar: Solid Dark */
        [data-testid="stSidebar"] {
            background-color: #020617; /* Even darker navy */
            border-right: 1px solid var(--card-border);
        }
        
        [data-testid="stSidebar"] * {
            color: var(--text-main) !important;
        }

        /* Tabs Optimization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            color: var(--text-dim);
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            color: var(--accent) !important;
            border-bottom-color: var(--accent) !important;
        }

        /* Typography */
        h1 {
            color: var(--text-main);
            font-weight: 800;
            font-size: 2.5rem !important;
            letter-spacing: -0.025em;
        }
        
        .subtitle {
            color: var(--text-dim);
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 2rem;
        }

        /* Result Cards: Dark Surface */
        .result-card {
            background: var(--secondary-bg);
            border-radius: 12px;
            border: 1px solid var(--card-border);
            padding: 30px;
            margin-bottom: 25px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 16px;
        }

        .status-sla { 
            background-color: rgba(16, 185, 129, 0.15); 
            color: var(--success);
            border: 1px solid var(--success);
        }
        
        .status-no-sla { 
            background-color: rgba(251, 113, 133, 0.15); 
            color: var(--error);
            border: 1px solid var(--error);
        }

        /* Progress Bar */
        .confidence-label {
            font-size: 0.875rem;
            color: var(--text-dim);
            font-weight: 600;
            margin-bottom: 4px;
        }
        
        .confidence-value {
            font-size: 2.25rem;
            font-weight: 800;
            color: var(--text-main);
            margin-bottom: 12px;
        }

        .bar-container {
            background: rgba(255,255,255,0.05);
            border-radius: 999px;
            height: 12px;
            width: 100%;
            overflow: hidden;
            border: 1px solid var(--card-border);
        }
        
        .bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        /* Inputs & Textareas */
        .stTextArea textarea {
            background-color: #0f172a !important;
            color: white !important;
            border: 1px solid var(--card-border) !important;
        }

        /* Buttons: Neon/Accent feel */
        .stButton > button {
            background-color: var(--accent) !important;
            color: #020617 !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            padding: 0.75rem 2rem !important;
            border: none !important;
            box-shadow: 0 4px 14px 0 rgba(45, 212, 191, 0.39) !important;
        }
        
        .stButton > button:hover {
            transform: scale(1.02);
            box-shadow: 0 6px 20px rgba(45, 212, 191, 0.45) !important;
        }

        </style>
    """, unsafe_allow_html=True)

def display_header(title, subtitle=None):
    """Displays a dark-mode header."""
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='subtitle'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 1px; background: var(--card-border); margin-bottom: 2.5rem;'></div>", unsafe_allow_html=True)

def display_result_card(prediction, probability, image_array=None):
    """Displays a high-contrast dark result card."""
    is_sla = prediction == 1
    status_class = "status-sla" if is_sla else "status-no-sla"
    status_text = "Detección Positiva: SLA" if is_sla else "Detección Negativa: No SLA"
    icon = "🧬" if is_sla else "🔬"
    bar_color = "var(--success)" if is_sla else "var(--error)"
    
    html_content = f"""
    <div class="result-card">
        <div class="status-badge {status_class}">{icon} {status_text}</div>
        <div class="confidence-label">Confianza Estructural</div>
        <div class="confidence-value">{probability:.1%}</div>
        <div class="bar-container">
            <div class="bar-fill" style="width: {probability*100}%; background-color: {bar_color};"></div>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)
    
    if image_array is not None:
        with st.expander("Ver Mapa de Características (RNN-FM Embedding)", expanded=True):
            st.image(image_array, use_container_width=True)

def plot_heatmap(df):
    """Plots an optimized heatmap for DARK MODE."""
    if "Tamaño Ventana" not in df.columns:
        st.warning("Datos insuficientes para el mapa de calor.")
        return

    # Process data for heatmap
    pivot_df = df.pivot(index='Tamaño Ventana', columns='Posición', values='Probabilidad SLA').sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        pivot_df.values, 
        aspect='auto', 
        cmap='magma', # More premium feel than YlOrRd
        vmin=0, vmax=1, 
        origin='lower',
        extent=[df['Posición'].min(), df['Posición'].max(), pivot_df.index.min(), pivot_df.index.max()]
    )
    
    ax.set_title("Frecuencia Estructural Multi-Escala", fontsize=12, pad=15, fontweight='bold')
    ax.set_xlabel("Posición Genómica (nucleótidos)", fontsize=10)
    ax.set_ylabel("Escala de Ventana (nt)", fontsize=10)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probabilidad SLA', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
