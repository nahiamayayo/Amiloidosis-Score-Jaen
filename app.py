import streamlit as st
import pandas as pd
import re
import pdfplumber
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo

# Configuración inicial de la página (DEBE ser la primera línea de Streamlit)
st.set_page_config(page_title="Amiloidosis-Score", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

# --- INYECCIÓN DE CSS PERSONALIZADO (DISEÑO CLÍNICO) ---
st.markdown("""
<style>
    /* Fondo general más limpio */
    .stApp {
        background-color: #f8f9fa;
    }
    /* Estilo de tarjeta para las métricas (Score) */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    /* Títulos institucionales */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Alertas de Streamlit más redondeadas */
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- BARRA LATERAL (SIDEBAR) INSTITUCIONAL ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=80) # Icono genérico médico (puedes cambiarlo por el logo de tu uni/hospital)
    st.title("Unidad de Cardiología")
    st.markdown("---")
    st.markdown("**Sistema Inteligente de Cribado**")
    st.markdown("Proyecto de validación retrospectiva para la detección temprana de Amiloidosis Cardíaca mediante analítica de rutina.")
    st.markdown("---")
    st.caption("Versión: 1.0 (Prototipo Clínico)")

# --- CABECERA PRINCIPAL ---
st.title("🫀 Amiloidosis-Score-Jaén")
st.markdown("Bienvenido al sistema automatizado de triaje. Sube un informe de laboratorio para iniciar la evaluación del paciente.")

# --- PESTAÑAS ---
tab1, tab2 = st.tabs(["📋 Evaluación Individual", "📊 Validación por Lotes (Cohorte)"])

with tab1:
    st.subheader("Paso 1: Extracción Automática")
    
    # Contenedor visual para la subida del PDF
    with st.container():
        pdf_subido = st.file_uploader("Arrastra aquí el PDF de la analítica del SAS", type=["pdf"])
    
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
            
            st.success(f"✅ Extracción completada: {variables_encontradas}/12 parámetros localizados.")
        except Exception as e:
            st.error(f"Error al leer el PDF: {e}")

    # Acordeón para la edición manual (Mantiene la UI limpia)
    with st.expander("🛠️ Ver / Editar parámetros extraídos manualmente", expanded=(pdf_subido is None)):
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
    
    # Botón centrado y destacado
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        calcular = st.button("🧬 Ejecutar Algoritmo de Riesgo", type="primary", use_container_width=True)

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
            
            st.markdown("### 📋 Informe de Resultados")
            
            # Distribución visual de resultados
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric(label="Score Logístico (Puntuación Bruta)", value=score)
                st.caption("Pendiente de calibración de corte local.")
                
            with res_col2:
                st.markdown("#### Perfil de Riesgo Clínico")
                if len(alertas) >= 3:
                    st.error(f"🚨 **ALTA SOSPECHA:** {len(alertas)} de 5 marcadores en riesgo.")
                elif len(alertas) > 0:
                    st.warning(f"⚠️ **RIESGO MODERADO:** {len(alertas)} marcadores en riesgo.")
                else:
                    st.success("🟢 **BAJO RIESGO:** Patrón bioquímico normal.")
                
                if alertas:
                    for alerta in alertas:
                        st.markdown(f"🔸 {alerta}")

with tab2:
    st.header("Validación Retrospectiva de Cohorte")
    st.markdown("Sube el archivo Excel `.xlsx` o `.csv` anonimizado con la cohorte local para calibrar la regresión logística.")
    archivo_subido = st.file_uploader("Selecciona la base de datos de pacientes", type=["xlsx", "csv"])
    
    if archivo_subido is not None:
        st.success("Base de datos en memoria. Lista para el análisis.")
        if st.button("🚀 Iniciar Calibración de Modelo"):
            st.info("Módulo de aprendizaje en desarrollo.")
