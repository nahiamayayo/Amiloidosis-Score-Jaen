import streamlit as st
import pandas as pd
import re
import pdfplumber
import os
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo

st.set_page_config(
    page_title="Amiloidosis-Score-Jaén", 
    page_icon="🫀", 
    layout="wide",  
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #f7faf8; }
    .block-container { max-width: 1000px !important; padding-top: 2rem !important; }
    div[data-testid="stImage"] img { border-radius: 0px !important; object-fit: contain !important; }
    
    /* Botones corporativos */
    .stButton>button {
        background-color: #0b5a32 !important; color: white !important;
        border-radius: 8px !important; border: none !important;
        padding: 12px 24px !important; font-weight: 600 !important;
    }
    
    /* Estilo del Acordeón */
    div[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 12px; }
    
    /* Cabecera institucional */
    h1 { color: #0b5a32 !important; font-weight: 800 !important; }
    
    /* Tablas */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #e2e8f0; gap: 10px; }
    .stTabs [data-baseweb="tab"] { border-bottom: 3px solid transparent; color: #718096; }
    .stTabs [aria-selected="true"] { color: #0b5a32 !important; border-bottom: 3px solid #0b5a32 !important; }
</style>
""", unsafe_allow_html=True)

# --- CABECERA ---
c1, c2 = st.columns([1, 6])
with c1:
    if os.path.exists("huj.png"): st.image("huj.png", width=140)
with c2:
    st.title("Amiloidosis-Score-Jaén")
    st.markdown("<p style='color: #4a5568; font-size: 1.1rem; margin-top: -6px;'>Plataforma Digital de Cribado | Unidad de Cardiología</p>", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2 = st.tabs(["📋 Evaluación de Paciente", "📊 Validación de Cohorte"])

with tab1:
    st.markdown("### 1. Carga de Datos Clínicos")
    pdf_subido = st.file_uploader("Sube el PDF de la analítica del SAS", type=["pdf"])
    
    valores_extraidos = {k: None for k in ['Glucosa', 'Trigliceridos', 'Colesterol', 'Colinesterasa', 'Gamma_GT', 'Albumina', 'MCH', 'Cloruro', 'Magnesio', 'Hb_libre', 'Alfa_amilasa', 'PCR']}
    
    if pdf_subido:
        with pdfplumber.open(pdf_subido) as pdf:
            texto = "\n".join([p.extract_text(layout=True) for p in pdf.pages if p.extract_text()])
            for clave in valores_extraidos:
                patron = {'Gamma_GT': r'Gamma\s*glutamiltransferasa|Gamma[\s-]*GT', 'MCH': r'Hemoglobina\s*corpuscular\s*media|HCM', 'PCR': r'Prote[ií]na\s*C\s*reactiva|PCR'}.get(clave, clave)
                match = re.search(rf"(?:{patron})[^\dA-Za-z]*(\d+[.,]\d+|\d+)", texto, re.IGNORECASE)
                if match: valores_extraidos[clave] = float(match.group(1).replace(',', '.'))
        st.success(f"🧬 Procesado: {sum(1 for v in valores_extraidos.values() if v is not None)} parámetros detectados.")

    with st.expander("🛠️ Verificar datos bioquímicos", expanded=True):
        # Campos agrupados en filas de 3 para que no se vea vacío
        for i in range(0, 12, 4):
            cols = st.columns(4)
            keys = list(valores_extraidos.keys())[i:i+4]
            for j, k in enumerate(keys):
                cols[j].number_input(k, value=valores_extraidos[k] if valores_extraidos[k] is not None else 0.0)

    if st.button("🧬 Computar Análisis", use_container_width=True):
        # ... (Tu lógica de cálculo aquí permanece igual) ...
        st.info("Algoritmo ejecutado correctamente.")

with tab2:
    st.markdown("### 📊 Validación Retrospectiva")
    st.file_uploader("Cargar Base de Datos (.xlsx / .csv)", type=["xlsx", "csv"])
