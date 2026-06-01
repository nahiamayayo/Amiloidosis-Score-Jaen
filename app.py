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
from xgboost import XGBClassifier

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
# INFORME PARA HISTORIA CLÍNICA
                st.write("")
                with st.expander("📝 Copiar Informe para Historia Clínica (Diraya)"):
                    st.code(f"""========================================================
INFORME DE ESTRATIFICACIÓN - PROTOCOLO CARDIOGEN JAÉN
========================================================

RESULTADO DEL ANÁLISIS MULTIVARIANTE (MACHINE LEARNING):
- Probabilidad predictiva del algoritmo: {probabilidad:.1%}
- Punto de corte de seguridad (institucional): {umbral_clinico:.1%}
- CATEGORIZACIÓN DE RIESGO CLÍNICO: {riesgo_texto}

PERFIL DE BIOMARCADORES DESTACADOS:
- NT-proBNP: {valores_extraidos['proBNP']} pg/mL
- Albúmina sérica: {valores_extraidos['Albumina']} g/dL
- Magnesio sérico: {valores_extraidos['Magnesio']} mg/dL

* NOTA TÉCNICA: La probabilidad ha sido calculada evaluando
la firma bioquímica completa de 10 parámetros de rutina. 
Los valores no disponibles en la analítica primaria han sido 
estimados de forma automatizada mediante imputación estadística 
(mediana poblacional de la cohorte local) para asegurar la 
validez predictiva del modelo.
========================================================""", language="text")

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
            st.error(f"Error procesando el Excel. Detalle: {e}")
            return None, None, None
    return None, None, None

with tab2:
    st.markdown("### 🧠 Calibración del Modelo Lineal (Regresión Logística)")
    st.info("Entrena el modelo base y extrae los coeficientes matemáticos exactos para la documentación clínica.")
    X, y, cols_finales = preparar_datos_csv()
    
    if X is not None and st.button("🚀 Iniciar Calibración", use_container_width=True):
        imputer = SimpleImputer(strategy='median')
        X_imp = imputer.fit_transform(X)
        scaler = StandardScaler()
        X_sca = scaler.fit_transform(X_imp)
        
        clf = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf.fit(X_sca, y)
        
        umbral_actual = st.session_state.get('umbral_jaen', 0.25)
        
        with open(ruta_modelo, 'wb') as archivo:
            pickle.dump({'clf': clf, 'imputer': imputer, 'scaler': scaler, 'columnas': cols_finales, 'umbral': umbral_actual}, archivo)
            
        st.session_state.update({'modelo_entrenado': True, 'clf_jaen': clf, 'imputer_jaen': imputer, 'scaler_jaen': scaler, 'columnas_jaen': cols_finales})
        st.success("✅ Motor lineal calibrado y guardado en el servidor.")
        
        st.markdown("#### 📐 Coeficientes Matemáticos Exactos")
        st.markdown("Copia estos valores para tu documento de validación. Un valor negativo indica que la disminución del parámetro suma riesgo (ej. Glucosa); un valor positivo indica que el aumento suma riesgo.")
        
        # Extracción exacta de coeficientes
        df_coef = pd.DataFrame({
            'Biomarcador': cols_finales,
            'Coeficiente (Peso)': clf.coef_[0]
        }).sort_values('Coeficiente (Peso)', ascending=False)
        
        st.dataframe(df_coef, use_container_width=True)

with tab3:
    st.markdown("### 📊 Auditoría Clínica y Comparativa Algorítmica")
    st.info("Módulo de validación interna mediante partición aleatoria (Hold-out 75/25). Evalúa el rendimiento diagnóstico de la Regresión Logística Multivariante frente a modelos de aprendizaje automático no lineal (Gradient Tree Boosting) para optimizar el punto de corte institucional.")
    
    if X is not None:
        if st.button("Generar Informe Comparativo y Actualizar Umbral", use_container_width=True):
            # Split de datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
            
            # Preprocesamiento
            imp_val = SimpleImputer(strategy='median')
            sca_val = StandardScaler()
            X_train_sca = sca_val.fit_transform(imp_val.fit_transform(X_train))
            X_test_sca = sca_val.transform(imp_val.transform(X_test))
            
            # --- MODELO 1: LINEAL (LogReg) ---
            clf_lr = LogisticRegression(max_iter=2000, class_weight='balanced')
            clf_lr.fit(X_train_sca, y_train)
            y_pred_prob_lr = clf_lr.predict_proba(X_test_sca)[:, 1]
            fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_prob_lr)
            auc_lr = auc(fpr_lr, tpr_lr)
            
            # Umbral LogReg
            youden_idx = np.argmax(tpr_lr - fpr_lr)
            nuevo_umbral = thresholds_lr[youden_idx]
            
            # Actualizamos el umbral en la app
            if st.session_state['modelo_entrenado']:
                with open(ruta_modelo, 'rb') as archivo: d = pickle.load(archivo)
                d['umbral'] = nuevo_umbral
                with open(ruta_modelo, 'wb') as archivo: pickle.dump(d, archivo)
                st.session_state['umbral_jaen'] = nuevo_umbral
            
            # --- MODELO 2: NO LINEAL (XGBoost) ---
            from xgboost import XGBClassifier
            # Configuramos XGBoost para manejar el desbalanceo y los datos
            scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
            clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
            clf_xgb.fit(X_train_sca, y_train)
            y_pred_prob_xgb = clf_xgb.predict_proba(X_test_sca)[:, 1]
            fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob_xgb)
            auc_xgb = auc(fpr_xgb, tpr_xgb)

            # --- RENDERIZADO DEL INFORME ---
            st.success(f"💾 Punto de Corte Clínico (Regresión Logística) fijado en: {nuevo_umbral*100:.1f}%")
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("AUC (Modelo Lineal)", f"{auc_lr:.3f}")
            col_m2.metric("AUC (Modelo XGBoost)", f"{auc_xgb:.3f}")
            
            # Gráfica Comparativa ROC
            st.markdown("#### 📈 Curvas ROC Comparativas")
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(fpr_lr, tpr_lr, color='#0b5a32', lw=2, label=f'Lineal (AUC = {auc_lr:.3f})')
            ax.plot(fpr_xgb, tpr_xgb, color='#e67e22', lw=2, linestyle='--', label=f'XGBoost (AUC = {auc_xgb:.3f})')
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':')
            ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
            ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
            ax.legend(loc="lower right")
            st.pyplot(fig)
            
    else:
        st.warning("Carga el archivo Excel arriba para realizar la auditoría.")
