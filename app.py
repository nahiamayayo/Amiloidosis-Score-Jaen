import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
st.set_page_config(page_title="Amiloidosis-Score-Jaén", page_icon="🫀", layout="wide", initial_sidebar_state="collapsed")

# Carga de la memoria (Persistencia del modelo y del umbral)
ruta_modelo = 'modelo_jaen.pkl'
if os.path.exists(ruta_modelo):
    with open(ruta_modelo, 'rb') as archivo:
        datos = pickle.load(archivo)
        st.session_state['modelo_entrenado'] = True
        st.session_state['clf_jaen'] = datos['clf']
        st.session_state['imputer_jaen'] = datos['imputer']
        st.session_state['scaler_jaen'] = datos['scaler']
        st.session_state['columnas_jaen'] = datos['columnas']
        st.session_state['umbral_jaen'] = datos.get('umbral', 0.25) # Si no hay umbral, usa 25% por defecto
else:
    st.session_state['modelo_entrenado'] = False
    st.session_state['umbral_jaen'] = 0.25

# Función de limpieza (Robusta para el Excel)
def limpiar_valor_para_entrenamiento(val):
    if pd.isna(val): return np.nan
    val_str = str(val).strip().upper().replace(',', '.')
    if val_str in ['NP', 'MHC', 'MNR', '-', 'MAR', '']: return np.nan
    if '<' in val_str or '>' in val_str: return float(re.sub(r'[<>]', '', val_str).strip())
    try: return float(val_str)
    except: return np.nan

# --- 2. ESTILOS VISUALES ---
st.markdown("""
<style>
    .stApp { background-color: #f7faf8; }
    .block-container { max-width: 1100px !important; padding-top: 2rem !important; }
    div[data-testid="stImage"] { display: flex; align-items: center; justify-content: center; }
    div[data-testid="stImage"] img { max-height: 85px !important; width: auto !important; }
    .stButton>button { background-color: #0b5a32 !important; color: white !important; font-weight: 600 !important; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #e2e8f0; }
    .stTabs [aria-selected="true"] { color: #0b5a32 !important; border-bottom: 3px solid #0b5a32 !important; font-weight: 700 !important; }
    h1, h2, h3 { color: #0b5a32 !important; font-family: 'Segoe UI', system-ui, sans-serif; font-weight: 700 !important; }
    div.row-widget.stRadio > div { flex-direction:row; justify-content: center; background-color: #e2e8f0; padding: 10px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 3. CABECERA ---
header_col1, header_col2 = st.columns([1.2, 6], vertical_alignment="center")
with header_col1:
    if os.path.exists("huj.png"): st.image("huj.png", use_container_width=True) 
with header_col2:
    st.title("Amiloidosis-Score-Jaén")
    st.markdown("<p style='color: #718096; font-size: 1.1rem; margin-top: -6px;'>Plataforma Digital de Cribado | Unidad de Cardiología</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋 Evaluación Clínica", "🧠 Entrenamiento de Base", "📊 Validación y Calibración"])

# ==========================================
# PESTAÑA 1: EVALUACIÓN EN CONSULTA
# ==========================================
with tab1:
    st.markdown("### 1. Extracción de Datos Clínicos")
    pdf_subido = st.file_uploader("Subir analítica (PDF)", type=["pdf"], label_visibility="collapsed")
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
            st.success("✅ Extracción digital completada.")
        except Exception as e: st.error(f"Error procesando PDF: {e}")

    with st.expander("🛠️ Revisión Manual de Valores", expanded=True):
        st.caption("Si un valor es 0.0, el algoritmo asumirá que no se realizó la prueba e imputará el valor medio del hospital.")
        cols = st.columns(2)
        for i, (k, v) in enumerate(valores_extraidos.items()):
            valores_extraidos[k] = cols[i % 2].number_input(k, value=float(v))
    
    st.markdown("### 2. Motor Predictivo")
    tipo_modelo = st.radio("Protocolo de Estratificación:", ["Score Local: CardioGen Jaén", "Score Base: Estudio Austríaco"])
    
    if st.button("🧬 Ejecutar Algoritmo", use_container_width=True):
        if tipo_modelo == "Score Base: Estudio Austríaco":
            resultado = procesar_analitica_paciente(valores_extraidos)
            if resultado['estado'] == "ERROR": st.error(resultado['mensaje'])
            else:
                score = calcular_score_bruto(resultado['datos_procesados'])
                alertas = evaluar_perfil_riesgo(resultado['datos_procesados'])
                col1, col2 = st.columns([1, 2])
                col1.metric("Score Bruto", score)
                with col2:
                    if alertas:
                        for alerta in alertas: st.warning(alerta)
                    else: st.success("Bajo Riesgo Clínico")
        else:
            if not st.session_state['modelo_entrenado']:
                st.error("⚠️ El Score Local no está configurado. Entrena el modelo en la Pestaña 2.")
            else:
                # PREPARACIÓN INTELIGENTE DE DATOS (Solución al problema de los Ceros)
                datos_paciente = []
                for col in st.session_state['columnas_jaen']:
                    val = valores_extraidos.get(col, 0.0)
                    datos_paciente.append(np.nan if val == 0.0 else val) # Si es 0, lo vaciamos para que el imputer trabaje
                
                # PIPELINE
                X_paciente = np.array(datos_paciente).reshape(1, -1)
                X_imputed = st.session_state['imputer_jaen'].transform(X_paciente)
                X_scaled = st.session_state['scaler_jaen'].transform(X_imputed)
                
                probabilidad = st.session_state['clf_jaen'].predict_proba(X_scaled)[0][1]
                umbral_clinico = st.session_state['umbral_jaen']
                
                # RESULTADOS CON SEMÁFORO DINÁMICO
                st.markdown("---")
                col_res1, col_res2 = st.columns([1, 2])
                col_res1.metric("Probabilidad Calculada", f"{probabilidad:.1%}")
                
                with col_res2:
                    if probabilidad >= umbral_clinico:
                        st.error(f"🚨 **Riesgo Significativo.** Supera el umbral de seguridad local ({umbral_clinico:.1%}). Se recomienda valoración específica.")
                        riesgo_texto = "ALTO"
                    else:
                        st.success(f"🟢 **Bajo Riesgo.** Por debajo del umbral de alarma ({umbral_clinico:.1%}).")
                        riesgo_texto = "BAJO"
                
                # INFORME PARA HISTORIA CLÍNICA
                st.write("")
                with st.expander("📝 Copiar para Historia Clínica (Diraya)"):
                    st.code(f"""VALORACIÓN PROTOCOLO CARDIOGEN JAÉN
-----------------------------------
Probabilidad de Amiloidosis: {probabilidad:.1%}
Umbral de corte institucional: {umbral_clinico:.1%}
Estratificación de Riesgo: {riesgo_texto}

Biomarcadores detectados:
- proBNP: {valores_extraidos['proBNP']}
- Albúmina: {valores_extraidos['Albumina']}
- Magnesio: {valores_extraidos['Magnesio']}
(Algoritmo con imputación automática de valores ausentes según mediana local).
""", language="text")

# ==========================================
# PESTAÑA 2 Y 3: ENTRENAMIENTO Y AUDITORÍA
# ==========================================
def preparar_datos_csv():
    archivo_csv = st.file_uploader("Cargar Base de Datos Histórica (.xlsx / .csv)", type=["xlsx", "csv"])
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
        try:
            X = df_limpio[columnas_finales].apply(lambda col: col.map(limpiar_valor_para_entrenamiento))
            y = df_limpio['Diagnóstico final']
            return X, y, columnas_finales
        except Exception as e:
            st.error(f"Error procesando el Excel: Asegúrate de tener la columna 'Diagnóstico final'. Detalle: {e}")
            return None, None, None
    return None, None, None

with tab2:
    st.markdown("### 🧠 Calibración Básica del Motor")
    st.info("Sube la base de datos para actualizar los coeficientes y las medianas de imputación.")
    X, y, cols_finales = preparar_datos_csv()
    
    if X is not None and st.button("🚀 Iniciar Calibración", use_container_width=True):
        imputer = SimpleImputer(strategy='median')
        X_imp = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_sca = scaler.fit_transform(X_imp)
        
        clf = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf.fit(X_sca, y)
        
        # Mantener el umbral que ya tuviéramos, o usar 25% si es la primera vez
        umbral_actual = st.session_state.get('umbral_jaen', 0.25)
        
        with open(ruta_modelo, 'wb') as archivo:
            pickle.dump({'clf': clf, 'imputer': imputer, 'scaler': scaler, 'columnas': cols_finales, 'umbral': umbral_actual}, archivo)
            
        st.session_state.update({'modelo_entrenado': True, 'clf_jaen': clf, 'imputer_jaen': imputer, 'scaler_jaen': scaler, 'columnas_jaen': cols_finales})
        st.success("✅ Motor calibrado y guardado. Los valores ausentes se imputarán ahora con la nueva base de datos.")

with tab3:
    st.markdown("### 📊 Auditoría y Definición del Umbral Clínico")
    st.info("Este módulo divide la base de datos (75/25) para calcular estadísticamente el Punto de Corte Óptimo (Índice de Youden).")
    if X is not None:
        if st.button("Generar Informe y Actualizar Umbral", use_container_width=True):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            
            imp_val = SimpleImputer(strategy='median')
            sca_val = StandardScaler()
            X_train_sca = sca_val.fit_transform(imp_val.fit_transform(X_train))
            
            clf_val = LogisticRegression(max_iter=2000, class_weight='balanced')
            clf_val.fit(X_train_sca, y_train)
            
            y_pred_prob = clf_val.predict_proba(sca_val.transform(imp_val.transform(X_test)))[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            
            # Cálculo del Umbral Óptimo
            youden_idx = np.argmax(tpr - fpr)
            nuevo_umbral = thresholds[youden_idx]
            
            # Actualizamos el umbral en la memoria persistente
            if st.session_state['modelo_entrenado']:
                with open(ruta_modelo, 'rb') as archivo: d = pickle.load(archivo)
                d['umbral'] = nuevo_umbral
                with open(ruta_modelo, 'wb') as archivo: pickle.dump(d, archivo)
                st.session_state['umbral_jaen'] = nuevo_umbral
                st.success(f"💾 El nuevo umbral del {nuevo_umbral:.1%} se ha enlazado automáticamente con la Pestaña 1.")
            
            st.markdown(f"#### 🎯 Nuevo Punto de Corte Clínico: **{nuevo_umbral*100:.1f}%**")
            st.metric("Área Bajo la Curva (AUC)", f"{auc(fpr, tpr):.3f}")
    else:
        st.warning("Carga el archivo Excel arriba para realizar la auditoría.")
