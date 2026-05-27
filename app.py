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
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo

# Configuración de la página
st.set_config = st.set_page_config(
    page_title="Amiloidosis-Score-Jaén", 
    page_icon="🫀", 
    layout="wide",  
    initial_sidebar_state="collapsed"
)

# Función de limpieza auxiliar para el modelo local
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
    div[data-testid="metric-container"] { background-color: #ffffff; border: 2px solid #e2e8f0; border-left: 6px solid #0b5a32; padding: 24px; border-radius: 12px; box-shadow: 0px 4px 12px rgba(0,0,0,0.03); }
    h1 { color: #0b5a32 !important; font-family: 'Segoe UI', system-ui, sans-serif; font-weight: 700 !important; margin-bottom: 4px !important; }
    h2, h3, h4 { color: #2d3748 !important; font-family: 'Segoe UI', system-ui, sans-serif; font-weight: 600 !important; }
    div[data-testid="stFileUploader"] { background-color: #ffffff; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

# --- CABECERA ---
header_col1, header_col2 = st.columns([1.2, 6], vertical_alignment="center")
with header_col1:
    if os.path.exists("huj.png"): st.image("huj.png", use_container_width=True) 
    else: st.warning("Logo")
with header_col2:
    st.title("Amiloidosis-Score-Jaén")
    st.markdown("<p style='color: #718096; font-size: 1.1rem; margin-top: -6px;'>Plataforma Digital de Cribado de Amiloidosis Cardíaca | Unidad de Cardiología</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📋 Evaluación de Paciente Individual", "🧠 Entrenamiento de Modelo Local"])

# --- PESTAÑA 1 ---
with tab1:
    st.markdown("### Carga de Datos Clínicos")
    pdf_subido = st.file_uploader("Subir analítica médica", type=["pdf"], label_visibility="collapsed")
    
    # 10 parámetros definidos
    parametros_jaen = ['Glucosa', 'Trigliceridos', 'Colesterol', 'Gamma_GT', 'Albumina', 'MCH', 'Magnesio', 'Hb_libre', 'PCR', 'proBNP']
    valores_extraidos = {k: 0.0 for k in parametros_jaen}
    
    if pdf_subido is not None:
        try:
            texto_completo = ""
            with pdfplumber.open(pdf_subido) as pdf:
                for pagina in pdf.pages:
                    texto = pagina.extract_text(layout=True)
                    if texto: texto_completo += texto + "\n"
            
            patrones_nombres = {
                'Glucosa': r'Glucosa',
                'Trigliceridos': r'Triglic[eé]ridos|Triglic',
                'Colesterol': r'Colesterol',
                'Gamma_GT': r'Gamma\s*glutamiltransferasa|Gamma[\s-]*GT|G\.G\.T',
                'Albumina': r'Alb[uú]mina',
                'MCH': r'Hemoglobina\s*corpuscular\s*media|HCM|MCH',
                'Magnesio': r'Magnesio',
                'Hb_libre': r'Hemoglobina(?!\s*glicosilada|\s*corpuscular)',
                'PCR': r'Prote[ií]na\s*C\s*reactiva|PCR',
                'proBNP': r'pro-p[eé]ptido\s*natriur[eé]tico\s*cerebral|proBNP|NT-proBNP'
            }

            for clave, patron in patrones_nombres.items():
                regex = rf"(?:{patron})[^\dA-Za-z]*(\d+[.,]\d+|\d+)"
                match = re.search(regex, texto_completo, re.IGNORECASE)
                if match: valores_extraidos[clave] = float(match.group(1).replace(',', '.'))
            
            st.success("✅ Extracción digital completada.")
        except Exception as e:
            st.error(f"Error procesando PDF: {e}")

    with st.expander("Ver y verificar valores bioquímicos", expanded=True):
        cols = st.columns(2)
        for i, (k, v) in enumerate(valores_extraidos.items()):
            valores_extraidos[k] = cols[i % 2].number_input(k, value=float(v))
    
    if st.button("🧬 Computar Análisis"):
        resultado = procesar_analitica_paciente(valores_extraidos)
        st.success(resultado['mensaje'])

# --- PESTAÑA 2 ---
with tab2:
    st.markdown("### 🧠 Entrenamiento del 'Score CardioGen Jaén'")
    archivo_csv = st.file_uploader("Cargar Base de Datos (.xlsx / .csv)", type=["xlsx", "csv"])
    
    if archivo_csv:
        df = pd.read_csv(archivo_csv) if archivo_csv.name.endswith('.csv') else pd.read_excel(archivo_csv)
        
        # Diccionario para traducir cualquier variación encontrada en tu Excel
        traductor = {
            'Glucosa': 'Glucosa', 'Gluosa': 'Glucosa', 
            'Triglicéridos': 'Trigliceridos', 'Trigliéridos': 'Trigliceridos',
            'Colesterol': 'Colesterol', 'olesterol': 'Colesterol',
            'Gamma-GT': 'Gamma_GT', 'Albúmina': 'Albumina',
            'MH': 'MCH', 'MHC': 'MCH', 'MCH': 'MCH',
            'Magnesio': 'Magnesio', 'Hemoglobina': 'Hb_libre',
            'PR': 'PCR', 'PCR': 'PCR',
            'proB': 'proBNP', 'proBNP': 'proBNP',
            'Diagnóstio final': 'Diagnóstico final'
        }
        
        # Renombramos las columnas
        df_limpio = df.rename(columns=traductor)
        columnas_finales = ['Glucosa', 'Trigliceridos', 'Colesterol', 'Gamma_GT', 'Albumina', 'MCH', 'Magnesio', 'Hb_libre', 'PCR', 'proBNP']
        
        if st.button("🚀 Entrenar y Analizar Rendimiento"):
            try:
                # AQUÍ ESTÁ EL CAMBIO: .map() en lugar de .applymap()
                X = df_limpio[columnas_finales].map(limpiar_valor_para_entrenamiento)
                y = df_limpio['Diagnóstico final']
                
                # Entrenamiento
                clf = LogisticRegression(max_iter=2000, class_weight='balanced')
                X_imputed = SimpleImputer(strategy='median').fit_transform(X)
                clf.fit(X_imputed, y)
                
                st.markdown("---")
                # 1. Estabilidad
                scores_cv = cross_val_score(clf, X_imputed, y, cv=5)
                st.metric("Estabilidad (Cross-Validation)", f"{scores_cv.mean():.2%}")
                
                # 2. Importancia
                st.markdown("### 🧬 Importancia de Variables")
                importancias = pd.DataFrame({'Variable': columnas_finales, 'Peso': np.abs(clf.coef_[0])}).sort_values('Peso', ascending=False)
                fig_bar, ax_bar = plt.subplots(); importancias.plot(kind='barh', x='Variable', y='Peso', ax=ax_bar, color='#0b5a32')
                st.pyplot(fig_bar)
                
                # 3. Calibración
                st.markdown("### ⚖️ Calibración")
                prob_pred = clf.predict_proba(X_imputed)[:, 1]
                prob_true, prob_pred_cal = calibration_curve(y, prob_pred, n_bins=5)
                fig_cal, ax_cal = plt.subplots(); ax_cal.plot(prob_pred_cal, prob_true, marker='o', color='#0b5a32')
                ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray')
                st.pyplot(fig_cal)
                
            except KeyError as e:
                st.error(f"Error: Asegúrate de que las columnas en el Excel coincidan. Columnas detectadas: {df_limpio.columns.tolist()}. Error: {e}")
