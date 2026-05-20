import streamlit as st
import pandas as pd
import re
import pdfplumber
import os
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo

# Configuración de la página: Cambiamos a 'wide' para que no se vea tan condensado
st.set_page_config(
    page_title="Amiloidosis-Score-Jaén", 
    page_icon="🫀", 
    layout="wide",  
    initial_sidebar_state="collapsed"
)

# --- CONFIGURACIÓN DE ESTILO CLÍNICO AVANZADO (CSS) ---
st.markdown("""
<style>
    /* Fondo general de la aplicación */
    .stApp {
        background-color: #f7faf8;
    }
    
    /* Controlar el ancho máximo para que sea amplio pero no infinito */
    .block-container {
        max-width: 1100px !important;
        padding-top: 2rem !important;
    }
    
    /* EVITAR RECORTE Y CONTROLAR ALTURA DEL LOGO (ACTUALIZADO) */
    div[data-testid="stImage"] {
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: visible !important;
    }
    div[data-testid="stImage"] img {
        border-radius: 0px !important;
        object-fit: contain !important;
        box-shadow: none !important;
        max-height: 85px !important;
        width: auto !important;
    }
    
    /* Personalización de los botones principales */
    .stButton>button {
        background-color: #0b5a32 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0px 4px 6px rgba(11, 90, 50, 0.15) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #084424 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0px 6px 12px rgba(11, 90, 50, 0.25) !important;
    }
    
    /* --- DISEÑO DE PESTAÑAS (Eliminando la línea roja) --- */
    div[data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: none;
        border-bottom: 3px solid transparent;
        padding: 10px 20px;
        font-weight: 500;
        color: #718096;
    }
    .stTabs [aria-selected="true"] {
        color: #0b5a32 !important;
        border-bottom: 3px solid #0b5a32 !important;
        background-color: transparent !important;
        font-weight: 700 !important;
    }

    /* Tarjeta flotante para el Score Médico */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 2px solid #e2e8f0;
        border-left: 6px solid #0b5a32;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.03);
    }
    
    /* Tipografías institucionales */
    h1 {
        color: #0b5a32 !important;
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-weight: 700 !important;
        margin-bottom: 4px !important;
    }
    h2, h3, h4 {
        color: #2d3748 !important;
        font-family: 'Segoe UI', system-ui, sans-serif;
        font-weight: 600 !important;
    }
    
    /* Estilización del área de arrastre del archivo PDF */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- CABECERA PRINCIPAL (DISEÑO AMPLIO Y ALINEADO) ---
# Ajustamos a [1.2, 6] para dar más espacio al logo y añadimos alineación vertical centrada
header_col1, header_col2 = st.columns([1.2, 6], vertical_alignment="center")

with header_col1:
    if os.path.exists("huj.png"):
        # Usamos el ancho del contenedor en lugar de píxeles fijos
        st.image("huj.png", use_container_width=True) 
    else:
        st.warning("Logo")

with header_col2:
    st.title("Amiloidosis-Score-Jaén")
    st.markdown("<p style='color: #718096; font-size: 1.1rem; margin-top: -6px;'>Plataforma Digital de Cribado de Amiloidosis Cardíaca | Unidad de Cardiología</p>", unsafe_allow_html=True)

# --- PESTAÑAS DE TRABAJO ---
tab1, tab2 = st.tabs(["📋 Evaluación de Paciente Individual", "📊 Validación Retrospectiva por Lotes"])

with tab1:
    st.write("")
    st.markdown("### Carga de Datos Clínicos")
    st.markdown("Sube la analítica de rutina anonimizada del paciente en formato PDF. El sistema analizará el documento de manera automatizada.")
    
    pdf_subido = st.file_uploader("Subir analítica médica", type=["pdf"], label_visibility="collapsed")

    valores_extraidos = {
        'Glucosa': None, 'Trigliceridos': None, 'Colesterol': None,
        'Colinesterasa': None, 'Gamma_GT': None, 'Albumina': None,
        'MCH': None, 'Cloruro': None, 'Magnesio': None,
        'Hb_libre': None, 'Alfa_amilasa': None, 'PCR': None
    }
    
    if pdf_subido is not None:
        try:
            texto_completo = ""
            with pdfplumber.open(pdf_subido) as pdf:
                for pagina in pdf.pages:
                    texto = pagina.extract_text(layout=True)
                    if texto:
                        texto_completo += texto + "\n"
            
            patrones_nombres = {
                'Glucosa': r'Glucosa',
                'Trigliceridos': r'Triglic[eé]ridos|Triglic',
                'Colesterol': r'Colesterol',
                'Colinesterasa': r'Colinesterasa',
                'Gamma_GT': r'Gamma\s*glutamiltransferasa|Gamma[\s-]*GT|G\.G\.T',
                'Albumina': r'Alb[uú]mina',
                'MCH': r'Hemoglobina\s*corpuscular\s*media|HCM',
                'Cloruro': r'Cloruro',
                'Magnesio': r'Magnesio',
                'Hb_libre': r'Hemoglobina(?!\s*glicosilada|\s*corpuscular)',
                'Alfa_amilasa': r'Alfa[\s-]*amilasa|Amilasa',
                'PCR': r'Prote[ií]na\s*C\s*reactiva|PCR'
            }

            variables_encontradas = 0
            for clave, patron in patrones_nombres.items():
                if valores_extraidos[clave] is None:
                    regex = rf"(?:{patron})[^\dA-Za-z]*(\d+[.,]\d+|\d+)"
                    match = re.search(regex, texto_completo, re.IGNORECASE)
                    if match:
                        valor_str = match.group(1).replace(',', '.')
                        valores_extraidos[clave] = float(valor_str)
                        variables_encontradas += 1
            
            st.success(f"🧬 Extracción digital completada: Se han localizado {variables_encontradas} de los 12 biomarcadores.")
        except Exception as e:
            st.error(f"Error en la lectura automatizada del documento: {e}")

    # Acordeón integrado para revisión manual (con más aire entre columnas)
    st.write("")
    with st.expander("Ver y verificar valores bioquímicos extraídos", expanded=(pdf_subido is None)):
        st.markdown("<p style='color: #4a5568; font-size: 0.95rem; margin-bottom: 15px;'>Modifica o introduce valores si el parámetro no constaba en el PDF original:</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3, gap="large") # gap="large" da mucho más espacio horizontal
        with col1:
            glucosa = st.number_input("Glucosa (mg/dL)", value=valores_extraidos['Glucosa'])
            trigliceridos = st.number_input("Triglicéridos (mg/dL)", value=valores_extraidos['Trigliceridos'])
            colesterol = st.number_input("Colesterol Total (mg/dL)", value=valores_extraidos['Colesterol'])
            cloruro = st.number_input("Cloruro (mmol/L)", value=valores_extraidos['Cloruro'])
        with col2:
            colinesterasa = st.number_input("Colinesterasa (kU/L)", value=valores_extraidos['Colinesterasa'])
            gamma_gt = st.number_input("Gamma-GT (U/L)", value=valores_extraidos['Gamma_GT'])
            albumina = st.number_input("Albúmina (g/L)", value=valores_extraidos['Albumina'])
            magnesio = st.number_input("Magnesio (mmol/L)", value=valores_extraidos['Magnesio'])
        with col3:
            mch = st.number_input("MCH (pg)", value=valores_extraidos['MCH'])
            hemoglobina_libre = st.number_input("Hb libre (µmol/L)", value=valores_extraidos['Hb_libre'])
            alfa_amilasa = st.number_input("Alfa-amilasa (U/L)", value=valores_extraidos['Alfa_amilasa'])
            pcr = st.number_input("PCR (mg/dL)", value=valores_extraidos['PCR'])

    st.write("")
    st.markdown("---")
    st.write("")
    
    # Botón centrado usando columnas vacías a los lados
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        calcular = st.button("🧬 Computar Análisis Computacional de Riesgo", use_container_width=True)

    if calcular:
        datos_paciente = {
            'Trigliceridos': trigliceridos, 'Glucosa': glucosa, 'Colinesterasa': colinesterasa,
            'Cloruro': cloruro, 'Albumina': albumina, 'Alfa_amilasa': alfa_amilasa,
            'PCR': pcr, 'Hb_libre': hemoglobina_libre, 'Magnesio': magnesio,
            'Gamma_GT': gamma_gt, 'MCH': mch, 'Colesterol': colesterol
        }
        
        resultado = procesar_analitica_paciente(datos_paciente)
        
        if resultado['estado'] == "ERROR":
            st.error(f"🛑 {resultado['mensaje']}")
        else:
            if "imputados" in resultado['mensaje']:
                st.info(f"ℹ️ {resultado['mensaje']}")
            else:
                st.success(f"✅ {resultado['mensaje']}")
            
            datos_limpios = resultado['datos_procesados']
            score = calcular_score_bruto(datos_limpios)
            alertas = evaluar_perfil_riesgo(datos_limpios)
            
            st.write("")
            st.markdown("### 📋 Informe Clínico de Cribado")
            
            res_col1, res_col2 = st.columns([1, 2], gap="large")
            
            with res_col1:
                st.metric(label="Score Logístico Bruto", value=score)
                st.caption("Puntuación matemática sujeta a calibración local.")
                
            with res_col2:
                st.markdown("#### Estratificación del Paciente")
                if len(alertas) >= 3:
                    st.error(f"🚨 **ALTA SOSPECHA BIOMÉDICA:** {len(alertas)} marcadores core alterados simultáneamente.")
                elif len(alertas) > 0:
                    st.warning(f"⚠️ **RIESGO MODERADO:** {len(alertas)} variables en rango de sospecha.")
                else:
                    st.success("🟢 **BAJO RIESGO:** Patrón bioquímico coincidente con control sano.")
                
                if alertas:
                    st.write("")
                    st.markdown("<p style='font-size: 0.95rem; font-weight: 600; margin-bottom: 8px;'>Marcadores de riesgo detectados:</p>", unsafe_allow_html=True)
                    for alerta in alertas:
                        st.markdown(f"• {alerta}")

with tab2:
    st.write("")
    st.markdown("### 📊 Calibración del Modelo (Ajuste Epidemiológico Local)")
    st.markdown("Carga la base de datos anonimizada del hospital (formato `.xlsx` o `.csv`) que contenga los históricos de pacientes confirmados y controles para realizar el ajuste multivariante de la constante matemática del algoritmo.")
    
    archivo_subido = st.file_uploader("Seleccionar base de datos de validación", type=["xlsx", "csv"])
    
    if archivo_subido is not None:
        st.success("Muestra de datos estructurada cargada en memoria con éxito.")
        st.write("")
        if st.button("Iniciar Calibración Multivariante y Ajuste ROC"):
            st.info("Módulo bioinformático en desarrollo.")
