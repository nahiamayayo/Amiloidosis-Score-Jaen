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
from sklearn.preprocessing import StandardScaler  # <--- ESTA ES LA LÍNEA QUE FALTABA
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo

# Configuración de la página
st.set_page_config(
    page_title="Amiloidosis-Score-Jaén", 
    page_icon="🫀", 
    layout="wide",  
    initial_sidebar_state="collapsed"
)

# Función de limpieza auxiliar robusta
def limpiar_valor_para_entrenamiento(val):
    if pd.isna(val): return np.nan
    val_str = str(val).strip().upper().replace(',', '.')
    if val_str in ['NP', 'MHC', 'MNR', '-', 'MAR', '']: return np.nan
    if '<' in val_str or '>' in val_str: return float(re.sub(r'[<>]', '', val_str).strip())
    try: return float(val_str)
    except: return np.nan

# --- CSS (Mantenido intacto) ---
st.markdown("""
<style>
    .stApp { background-color: #f7faf8; }
    .block-container { max-width: 1100px !important; padding-top: 2rem !important; }
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

# --- PESTAÑA 1 ---
with tab1:
    st.markdown("### Carga de Datos Clínicos")
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

    with st.expander("Ver y verificar valores", expanded=True):
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
        traductor = {
            'Gluosa': 'Glucosa', 'Glucosa': 'Glucosa', 'Trigliéridos': 'Trigliceridos', 'Triglicéridos': 'Trigliceridos',
            'olesterol': 'Colesterol', 'Colesterol': 'Colesterol', 'Gamma-GT': 'Gamma_GT', 'Albúmina': 'Albumina',
            'MH': 'MCH', 'MHC': 'MCH', 'MCH': 'MCH', 'Magnesio': 'Magnesio', 'Hemoglobina': 'Hb_libre',
            'PR': 'PCR', 'PCR': 'PCR', 'proB': 'proBNP', 'proBNP': 'proBNP',
            'Diagnóstio final': 'Diagnóstico final', 'Diagnóstico final': 'Diagnóstico final'
        }
        df_limpio = df.rename(columns=traductor)
        columnas_finales = ['Glucosa', 'Trigliceridos', 'Colesterol', 'Gamma_GT', 'Albumina', 'MCH', 'Magnesio', 'Hb_libre', 'PCR', 'proBNP']
        
        if st.button("🚀 Entrenar y Analizar"):
            try:
                X = df_limpio[columnas_finales].apply(lambda col: col.map(limpiar_valor_para_entrenamiento))
                y = df_limpio['Diagnóstico final']
                
                imputer = SimpleImputer(strategy='median')
                X_imputed = imputer.fit_transform(X)
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_imputed)
                
                clf = LogisticRegression(max_iter=2000, class_weight='balanced')
                clf.fit(X_scaled, y)
                
                st.metric("Estabilidad (Cross-Validation)", f"{cross_val_score(clf, X_scaled, y, cv=5).mean():.2%}")
                
                st.markdown("### 🧬 Importancia de Variables")
                importancias = pd.DataFrame({'Variable': columnas_finales, 'Peso': np.abs(clf.coef_[0])}).sort_values('Peso', ascending=False)
                fig, ax = plt.subplots(); importancias.plot(kind='barh', x='Variable', y='Peso', ax=ax, color='#0b5a32')
                st.pyplot(fig)
                
                st.markdown("### ⚖️ Calibración")
                prob_pred = clf.predict_proba(X_scaled)[:, 1]
                prob_true, prob_pred_cal = calibration_curve(y, prob_pred, n_bins=5)
                fig_cal, ax_cal = plt.subplots(); ax_cal.plot(prob_pred_cal, prob_true, marker='o', color='#0b5a32')
                ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray')
                st.pyplot(fig_cal)
            except Exception as e:
                st.error(f"Error: {e}")
