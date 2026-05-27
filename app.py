import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo

# Configuración de la página
st.set_page_config(
    page_title="Amiloidosis-Score-Jaén", 
    page_icon="🫀", 
    layout="wide",  
    initial_sidebar_state="collapsed"
)

# --- CONFIGURACIÓN DE ESTILO CLÍNICO AVANZADO (CSS) ---
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

# --- CABECERA PRINCIPAL ---
header_col1, header_col2 = st.columns([1.2, 6], vertical_alignment="center")

with header_col1:
    if os.path.exists("huj.png"):
        st.image("huj.png", use_container_width=True) 
    else:
        st.warning("Logo")

with header_col2:
    st.title("Amiloidosis-Score-Jaén")
    st.markdown("<p style='color: #718096; font-size: 1.1rem; margin-top: -6px;'>Plataforma Digital de Cribado de Amiloidosis Cardíaca | Unidad de Cardiología</p>", unsafe_allow_html=True)

# --- PESTAÑAS DE TRABAJO ---
tab1, tab2 = st.tabs(["📋 Evaluación de Paciente Individual", "🧠 Entrenamiento de Modelo Local"])

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
                'Glucosa': r'Glucosa', 'Trigliceridos': r'Triglic[eé]ridos|Triglic', 'Colesterol': r'Colesterol',
                'Colinesterasa': r'Colinesterasa', 'Gamma_GT': r'Gamma\s*glutamiltransferasa|Gamma[\s-]*GT|G\.G\.T',
                'Albumina': r'Alb[uú]mina', 'MCH': r'Hemoglobina\s*corpuscular\s*media|HCM', 'Cloruro': r'Cloruro',
                'Magnesio': r'Magnesio', 'Hb_libre': r'Hemoglobina(?!\s*glicosilada|\s*corpuscular)',
                'Alfa_amilasa': r'Alfa[\s-]*amilasa|Amilasa', 'PCR': r'Prote[ií]na\s*C\s*reactiva|PCR'
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

    st.write("")
    with st.expander("Ver y verificar valores bioquímicos extraídos", expanded=(pdf_subido is None)):
        st.markdown("<p style='color: #4a5568; font-size: 0.95rem; margin-bottom: 15px;'>Modifica o introduce valores si el parámetro no constaba en el PDF original:</p>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3, gap="large") 
        with col1:
            glucosa = st.number_input("Glucosa (mg/dL)", value=valores_extraidos['Glucosa'] if valores_extraidos['Glucosa'] is not None else 0.0)
            trigliceridos = st.number_input("Triglicéridos (mg/dL)", value=valores_extraidos['Trigliceridos'] if valores_extraidos['Trigliceridos'] is not None else 0.0)
            colesterol = st.number_input("Colesterol Total (mg/dL)", value=valores_extraidos['Colesterol'] if valores_extraidos['Colesterol'] is not None else 0.0)
            cloruro = st.number_input("Cloruro (mmol/L)", value=valores_extraidos['Cloruro'] if valores_extraidos['Cloruro'] is not None else 0.0)
        with col2:
            colinesterasa = st.number_input("Colinesterasa (kU/L)", value=valores_extraidos['Colinesterasa'] if valores_extraidos['Colinesterasa'] is not None else 0.0)
            gamma_gt = st.number_input("Gamma-GT (U/L)", value=valores_extraidos['Gamma_GT'] if valores_extraidos['Gamma_GT'] is not None else 0.0)
            albumina = st.number_input("Albúmina (g/L)", value=valores_extraidos['Albumina'] if valores_extraidos['Albumina'] is not None else 0.0)
            magnesio = st.number_input("Magnesio (mmol/L)", value=valores_extraidos['Magnesio'] if valores_extraidos['Magnesio'] is not None else 0.0)
        with col3:
            mch = st.number_input("MCH (pg)", value=valores_extraidos['MCH'] if valores_extraidos['MCH'] is not None else 0.0)
            hemoglobina_libre = st.number_input("Hb libre (µmol/L)", value=valores_extraidos['Hb_libre'] if valores_extraidos['Hb_libre'] is not None else 0.0)
            alfa_amilasa = st.number_input("Alfa-amilasa (U/L)", value=valores_extraidos['Alfa_amilasa'] if valores_extraidos['Alfa_amilasa'] is not None else 0.0)
            pcr = st.number_input("PCR (mg/dL)", value=valores_extraidos['PCR'] if valores_extraidos['PCR'] is not None else 0.0)

    st.write("")
    st.markdown("---")
    st.write("")
    
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
                st.caption("Score calculado en base al modelo original.")
                
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
    st.markdown("### 🧠 Creación de Score Local (Regresión Logística)")
    st.markdown("Entrena un nuevo modelo matemático usando tu base de datos para generar coeficientes específicos para la unidad.")
    
    archivo_csv = st.file_uploader("Seleccionar base de datos clínica (.xlsx / .csv)", type=["xlsx", "csv"])
    
    def limpiar_valor_para_entrenamiento(val):
        if pd.isna(val): return np.nan
        val_str = str(val).strip().upper().replace(',', '.')
        if val_str in ['NP', 'MHC', 'MNR', '-', 'MAR', '']: return np.nan
        if '<' in val_str or '>' in val_str: return float(re.sub(r'[<>]', '', val_str).strip())
        try: return float(val_str)
        except: return np.nan

    if archivo_csv is not None:
        try:
            if archivo_csv.name.endswith('.csv'): df = pd.read_csv(archivo_csv)
            else: df = pd.read_excel(archivo_csv)
                
            st.success(f"✅ Base de datos cargada: {len(df)} pacientes listos para entrenamiento.")
            st.write("")
            
            if st.button("🚀 Entrenar Score Local y Calcular Rendimiento", use_container_width=True):
                # 1. Mapeo adaptado a las columnas que existen en tu Excel
                mapa_columnas = {
                    'Glucosa': 'Glucosa', 'Gluosa': 'Glucosa', 
                    'Triglicéridos': 'Trigliceridos', 'Trigliéridos': 'Trigliceridos', 
                    'Colesterol': 'Colesterol', 'olesterol': 'Colesterol',
                    'Gamma-GT': 'Gamma_GT', 
                    'Albúmina': 'Albumina', 
                    'MCH': 'MCH', 'MH': 'MCH', 'MHC': 'MCH',
                    'Magnesio': 'Magnesio', 
                    'Hemoglobina': 'Hb_libre', 
                    'PCR': 'PCR', 'PR': 'PCR'
                }
                
                parametros_en_excel = list(set(mapa_columnas.values()))
                
                # 2. Extracción y limpieza estructurada para el algoritmo
                datos_limpios = []
                col_diagnostico = 'Diagnóstio final' if 'Diagnóstio final' in df.columns else 'Diagnóstico final'
                
                for index, row in df.iterrows():
                    fila_dict = {}
                    for col_excel in df.columns:
                        if col_excel in mapa_columnas:
                            col_motor = mapa_columnas[col_excel]
                            fila_dict[col_motor] = limpiar_valor_para_entrenamiento(row[col_excel])
                    fila_dict['Diagnostico'] = int(row[col_diagnostico])
                    datos_limpios.append(fila_dict)
                
                df_limpio = pd.DataFrame(datos_limpios)
                
                # 3. Preparación de datos (Machine Learning Pipeline)
                X = df_limpio.drop('Diagnostico', axis=1)
                y = df_limpio['Diagnostico']
                
                # Imputación: Rellenamos valores vacíos (NP) con la mediana de NUESTRA propia muestra
                imputer = SimpleImputer(strategy='median')
                X_imputed = imputer.fit_transform(X)
                
                # 4. Entrenamiento del Modelo (Regresión Logística como en el estudio)
                clf = LogisticRegression(max_iter=2000, class_weight='balanced')
                clf.fit(X_imputed, y)
                
                # 5. Generación del "Score Local"
                predicciones_prob = clf.predict_proba(X_imputed)[:, 1]
                coeficientes = clf.coef_[0]
                intercepto = clf.intercept_[0]
                
                st.markdown("---")
                st.markdown("### 🧬 El Nuevo Score CardioGen")
                st.markdown("El modelo ha calculado sus propios pesos basados en los patrones de los pacientes andaluces.")
                
                # Mostrar la ecuación matemática
                ecuacion = f"**Score** = {intercepto:.4f} "
                for i, col in enumerate(X.columns):
                    signo = "+" if coeficientes[i] >= 0 else "-"
                    ecuacion += f"{signo} ({abs(coeficientes[i]):.4f} × {col}) "
                
                st.info(ecuacion)
                
                # 6. Evaluación de Curva ROC
                if len(set(y)) > 1:
                    fpr, tpr, thresholds = roc_curve(y, predicciones_prob)
                    roc_auc = auc(fpr, tpr)
                    
                    youden_index = tpr - fpr
                    best_threshold_idx = youden_index.argmax()
                    best_threshold = thresholds[best_threshold_idx]
                    sensibilidad = tpr[best_threshold_idx]
                    especificidad = 1 - fpr[best_threshold_idx]
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Variables Entrenadas", len(X.columns))
                    c2.metric("Área bajo la curva (AUC)", f"{roc_auc:.3f}")
                    c3.metric("Sensibilidad Máxima", f"{sensibilidad*100:.1f}%")
                    
                    # Gráfica ROC
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, color='#0b5a32', lw=2, label=f'Score Local (AUC = {roc_auc:.3f})')
                    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                    ax.scatter(fpr[best_threshold_idx], tpr[best_threshold_idx], color='red', s=100, label=f'Punto Óptimo: Prob > {best_threshold:.2f}', zorder=5)
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('1 - Especificidad (Falsos Positivos)')
                    ax.set_ylabel('Sensibilidad (Verdaderos Positivos)')
                    ax.set_title('Rendimiento del Nuevo Score Entrenado con Datos de Jaén')
                    ax.legend(loc="lower right")
                    ax.grid(alpha=0.3)
                    
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ No se puede calcular la curva ROC (faltan controles o casos confirmados en la muestra).")
                    
        except Exception as e:
            st.error(f"Error procesando la base de datos para el entrenamiento: {e}")
