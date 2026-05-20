import streamlit as st
import pandas as pd
import re
import pdfplumber
import os
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo

# Configuración inicial de la página (Obligatoriamente la primera orden de Streamlit)
st.set_page_config(page_title="Amiloidosis-Score", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

# --- INYECCIÓN DE CSS PERSONALIZADO (DISEÑO CLÍNICO PROFESIONAL) ---
st.markdown("""
<style>
    /* Fondo general gris perla clínico */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Diseño de tarjeta flotante para el Score Bruto */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        text-align: center;
    }
    /* Títulos con tipografía limpia y color corporativo */
    h1, h2, h3 {
        color: #1e3d59;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    /* Suavizado de las alertas fijas */
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL INSTITUTIONAL CON TU LOGO ---
with st.sidebar:
    # Intenta cargar el logo local subido a GitHub; si no está, usa un respaldo limpio
    if os.path.exists("huj.png"):
        st.image("huj.png", use_container_width=True)
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=70)
        st.caption("⚠️ Sube 'huj.png' a GitHub para activar el logotipo oficial.")
    
    st.markdown("<h3 style='text-align: center; margin-top: 0;'>U. de Cardiología</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Cribado Bioinformático Inteligente**")
    st.markdown("Algoritmo de detección temprana de Amiloidosis Cardíaca mediante patrones analíticos de rutina.")
    st.markdown("---")
    st.caption("Hospital Universitario de Jaén\nVersión 1.1 Prototipo Clínico")

# --- CABECERA PRINCIPAL ---
st.title("🫀 Amiloidosis-Score-Jaén")
st.markdown("Herramienta de triaje automatizado orientada a optimizar la derivación diagnóstica.")

# --- PESTAÑAS DE TRABAJO ---
tab1, tab2 = st.tabs(["📋 Evaluación Individual", "📊 Validación por Lotes (Cohorte)"])

with tab1:
    st.subheader("Paso 1: Extracción Digital de Parámetros")
    
    # Módulo de carga de analíticas
    pdf_subido = st.file_uploader("Arrastra aquí el PDF anonimizado del laboratorio del SAS", type=["pdf"])
    
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
            
            # El motor de búsqueda global que logramos calibrar con éxito
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
            
            st.success(f"🧬 Extracción completada con éxito: {variables_encontradas}/12 parámetros localizados.")
        except Exception as e:
            st.error(f"Error en el procesamiento del PDF: {e}")

    # Panel de control de datos (Escondido ordenadamente en un acordeón)
    with st.expander("🛠️ Revisar / Modificar manualmente los valores del laboratorio", expanded=(pdf_subido is None)):
        col1, col2, col3 = st.columns(3)
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

    st.markdown("---")
    
    # Botón principal de ejecución
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        calcular = st.button("🧬 Analizar Perfil Bioquímico", type="primary", use_container_width=True)

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
            
            st.markdown("### 📋 Informe Estadístico y Clínico")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric(label="Score Logístico Bruto", value=score)
                st.caption("Puntuación lineal sujeta a ajuste por intercepto local.")
                
            with res_col2:
                st.markdown("#### Estratificación del Paciente")
                if len(alertas) >= 3:
                    st.error(f"🚨 **ALTA SOSPECHA BIOMÉDICA:** {len(alertas)} marcadores core alterados.")
                elif len(alertas) > 0:
                    st.warning(f"⚠️ **RIESGO MODERADO:** {len(alertas)} marcadores en ventana de riesgo.")
                else:
                    st.success("🟢 **BAJO RIESGO:** Firma bioquímica compatible con control sano.")
                
                if alertas:
                    for alerta in alertas:
                        st.markdown(f"🔸 {alerta}")

with tab2:
    st.header("Validación Retrospectiva de Cohorte")
    st.markdown("Sube el archivo Excel `.xlsx` o `.csv` anonimizado con la cohorte histórica del hospital para entrenar la constante de calibración local.")
    archivo_subido = st.file_uploader("Selecciona la base de datos de pacientes", type=["xlsx", "csv"])
    
    if archivo_subido is not None:
        st.success("Base de datos cargada en memoria. Registros listos para computar.")
        if st.button("🚀 Ejecutar Ajuste de Regresión Logística"):
            st.info("Módulo de calibración multivariante y generación de curva ROC en desarrollo.")
