import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo

# --- CONFIGURACIÓN E INICIALIZACIÓN DE MEMORIA ---
st.set_page_config(
    page_title="Amiloidosis-Score-Jaén", 
    page_icon="🫀", 
    layout="wide",  
    initial_sidebar_state="collapsed"
)

# Inicializamos la memoria de la app para guardar el modelo local
if 'modelo_entrenado' not in st.session_state:
    st.session_state['modelo_entrenado'] = False

# Función de limpieza auxiliar robusta
def limpiar_valor_para_entrenamiento(val):
    if pd.isna(val): return np.nan
    val_str = str(val).strip().upper().replace(',', '.')
    if val_str in ['NP', 'MHC', 'MNR', '-', 'MAR', '']: return np.nan
    if '<' in val_str or '>' in val_str: return float(re.sub(r'[<>]', '', val_str).strip())
    try: return float(val_str)
    except: return np.nan

# --- CSS ---
st.markdown("""
<style>
    .stApp { background-color: #f7faf8; }
    .block-container { max-width: 1100px !important; padding-top: 2rem !important; }
    div[data-testid="stImage"] { display: flex; align-items: center; justify-content: center; overflow: visible !important; }
    div[data-testid="stImage"] img { border-radius: 0px !important; object-fit: contain !important; box-shadow: none !important; max-height: 85px !important; width: auto !important; }
    .stButton>button { background-color: #0b5a32 !important; color: white !important; border-radius: 8px !important; border: none !important; padding: 10px 24px !important; font-weight: 600 !important; box-shadow: 0px 4px 6px rgba(11, 90, 50, 0.15) !important; transition: all 0.3s ease !important; }
    .stButton>button:hover { background-color: #084424 !important; transform: translateY(-1px) !important; box-shadow: 0px 6px 12px rgba(11, 90, 50, 0.25) !important; }
    div[data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: 2px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; border-bottom: 3px solid transparent; padding: 10px 20px; font-weight: 500; color: #718096; }
    .stTabs [aria-selected="true"] { color: #0b5a32 !important; border-bottom: 3px solid #0b5a32 !important; background-color: transparent !important; font-weight: 700 !important; }
    h1 { color: #0b5a32 !important; font-weight: 700 !important; margin-bottom: 4px !important; }
    
    /* Estilo para el selector de modelos */
    div.row-widget.stRadio > div { flex-direction:row; justify-content: center; background-color: #e2e8f0; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- CABECERA ---
header_col1, header_col2 = st.columns([1.2, 6], vertical_alignment="center")
with header_col1:
    if os.path.exists("huj.png"): st.image("huj.png", use_container_width=True) 
with header_col2:
    st.title("Amiloidosis-Score-Jaén")
    st.markdown("<p style='color: #718096; font-size: 1.1rem; margin-top: -6px;'>Plataforma Digital de Cribado | Unidad de Cardiología</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📋 Evaluación de Paciente Individual", "🧠 Entrenamiento de Modelo Local"])

# --- PESTAÑA 1: EVALUACIÓN ---
with tab1:
    st.markdown("### 1. Carga de Datos Clínicos")
    pdf_subido = st.file_uploader("Subir analítica médica", type=["pdf"], label_visibility="collapsed")
    parametros_jaen = ['Glucosa', 'Trigliceridos', 'Colesterol', 'Gamma_GT', 'Albumina', 'MCH', 'Magnesio', 'Hb_libre', 'PCR', 'proBNP']
    valores_extraidos = {k: 0.0 for k in parametros_jaen}
    
    if pdf_subido is not None:
        try:
            texto_completo = ""
            with pdfplumber.open(pdf_subido) as pdf:
                for pagina in pdf.pages:
                    t = pagina.extract_text(layout=True)
                    if t: texto_completo += t + "\n"
            
            patrones = {
                'Glucosa': r'Glucosa', 'Trigliceridos': r'Triglic[eé]ridos|Triglic', 'Colesterol': r'Colesterol',
                'Gamma_GT': r'Gamma\s*glutamiltransferasa|Gamma[\s-]*GT', 'Albumina': r'Alb[uú]mina',
                'MCH': r'Hemoglobina\s*corpuscular\s*media|HCM|MCH', 'Magnesio': r'Magnesio',
                'Hb_libre': r'Hemoglobina(?!\s*glicosilada|\s*corpuscular)',
                'PCR': r'Prote[ií]na\s*C\s*reactiva|PCR', 'proBNP': r'pro-p[eé]ptido\s*natriur[eé]tico\s*cerebral|proBNP|NT-proBNP'
            }
            for k, p in patrones.items():
                m = re.search(rf"(?:{p})[^\dA-Za-z]*(\d+[.,]\d+|\d+)", texto_completo, re.IGNORECASE)
                if m: valores_extraidos[k] = float(m.group(1).replace(',', '.'))
            st.success("✅ Extracción completada.")
        except Exception as e: st.error(f"Error PDF: {e}")

    with st.expander("🛠️ Ver y verificar valores", expanded=True):
        cols = st.columns(2)
        for i, (k, v) in enumerate(valores_extraidos.items()):
            valores_extraidos[k] = cols[i % 2].number_input(k, value=float(v))
    
    st.markdown("---")
    st.markdown("### 2. Motor Predictivo")
    
    # EL INTERRUPTOR CLÍNICO
    tipo_modelo = st.radio("Selecciona el algoritmo de estratificación:", 
                           ["Modelo Austríaco (Estudio Original)", "Modelo CardioGen Jaén (Calibrado Localmente)"])
    
    if st.button("🧬 Computar Análisis", use_container_width=True):
        if tipo_modelo == "Modelo Austríaco (Estudio Original)":
            # Usar el motor original estático
            resultado = procesar_analitica_paciente(valores_extraidos)
            if resultado['estado'] == "ERROR":
                st.error(resultado['mensaje'])
            else:
                st.info("Usando coeficientes y medianas del estudio base.")
                score = calcular_score_bruto(resultado['datos_procesados'])
                alertas = evaluar_perfil_riesgo(resultado['datos_procesados'])
                
                col1, col2 = st.columns([1, 2])
                col1.metric("Score Logístico Bruto", score)
                with col2:
                    if alertas:
                        for alerta in alertas: st.warning(alerta)
                    else: st.success("Bajo Riesgo Clínico")
                    
        else:
            # Usar el nuevo motor dinámico local
            if not st.session_state['modelo_entrenado']:
                st.error("⚠️ El Score CardioGen Jaén aún no está entrenado. Ve a la Pestaña 2, sube tu base de datos y entrena el modelo primero.")
            else:
                st.success("🧠 Usando el algoritmo entrenado localmente con datos de Jaén.")
                
                # Preparamos los datos del paciente para el modelo de ML
                datos_paciente = []
                for col in st.session_state['columnas_jaen']:
                    val = valores_extraidos.get(col, 0.0)
                    datos_paciente.append(np.nan if val == 0.0 else val)
                
                # Transformación: Imputar -> Escalar -> Predecir
                X_paciente = np.array(datos_paciente).reshape(1, -1)
                X_imputed = st.session_state['imputer_jaen'].transform(X_paciente)
                X_scaled = st.session_state['scaler_jaen'].transform(X_imputed)
                
                probabilidad = st.session_state['clf_jaen'].predict_proba(X_scaled)[0][1]
                
                col1, col2 = st.columns([1, 2])
                col1.metric("Probabilidad de Amiloidosis", f"{probabilidad:.1%}")
                
                with col2:
                    if probabilidad < 0.25:
                        st.success("🟢 Riesgo Bajo (< 25%). Control rutinario.")
                    elif probabilidad < 0.60:
                        st.warning("🟡 Riesgo Moderado (25% - 60%). Valoración cardiológica.")
                    else:
                        st.error("🚨 Riesgo Alto (> 60%). Evaluación urgente / Ecocardiograma.")

# --- PESTAÑA 2: ENTRENAMIENTO ---
with tab2:
    st.markdown("### 🧠 Entrenamiento del 'Score CardioGen Jaén'")
    st.info("Al entrenar el modelo aquí, se actualizará el algoritmo de la Pestaña 1 con los datos locales más recientes.")
    archivo_csv = st.file_uploader("Cargar Base de Datos (.xlsx / .csv)", type=["xlsx", "csv"])
    
    if archivo_csv:
        df = pd.read_csv(archivo_csv) if archivo_csv.name.endswith('.csv') else pd.read_excel(archivo_csv)
        traductor = {
            'Gluosa': 'Glucosa', 'Glucosa': 'Glucosa', 'Trigliéridos': 'Trigliceridos', 'Triglicéridos': 'Trigliceridos',
            'olesterol': 'Colesterol', 'Colesterol': 'Colesterol', 'Gamma-GT': 'Gamma_GT', 'Albúmina': 'Albumina',
            'MH': 'MCH', 'MHC': 'MCH', 'MCH': 'MCH', 'Magnesio': 'Magnesio', 'Hemoglobina': 'Hb_libre',
            'PR': 'PCR', 'PCR': 'PCR', 'proB': 'proBNP', 'proBNP': 'proBNP',
            'Diagnóstio final': 'Diagnóstico final', 'Diagnóstico final': 'Diagnóstico final'
        }
        df_limpio = df.rename(columns=traductor)
        columnas_finales = ['Glucosa', 'Trigliceridos', 'Colesterol', 'Gamma_GT', 'Albumina', 'MCH', 'Magnesio', 'Hb_libre', 'PCR', 'proBNP']
        
        if st.button("🚀 Entrenar Score Clínico Local", use_container_width=True):
            try:
                X = df_limpio[columnas_finales].apply(lambda col: col.map(limpiar_valor_para_entrenamiento))
                y = df_limpio['Diagnóstico final']
                
                imputer = SimpleImputer(strategy='median')
                X_imputed = imputer.fit_transform(X)
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_imputed)
                
                clf = LogisticRegression(max_iter=2000, class_weight='balanced')
                clf.fit(X_scaled, y)
                
                # GUARDAR EN MEMORIA (Session State)
                st.session_state['modelo_entrenado'] = True
                st.session_state['clf_jaen'] = clf
                st.session_state['imputer_jaen'] = imputer
                st.session_state['scaler_jaen'] = scaler
                st.session_state['columnas_jaen'] = columnas_finales
                
                st.success("✅ Modelo entrenado y guardado en memoria. Ya puedes usarlo en la Pestaña 1.")
                st.markdown("---")
                
                st.metric("Fiabilidad (Cross-Validation)", f"{cross_val_score(clf, X_scaled, y, cv=5).mean():.2%}")
                
                st.markdown("### 🧬 Importancia de Variables")
                importancias = pd.DataFrame({'Variable': columnas_finales, 'Peso': np.abs(clf.coef_[0])}).sort_values('Peso', ascending=False)
                fig, ax = plt.subplots(); importancias.plot(kind='barh', x='Variable', y='Peso', ax=ax, color='#0b5a32')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error: {e}")
