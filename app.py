import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import os
import pickle
import matplotlib.pyplot as plt
import altair as alt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
st.set_page_config(page_title="Protocolo CardioGen Jaén", layout="wide", initial_sidebar_state="collapsed")

ruta_modelo = 'modelo_jaen.pkl'
ruta_datos_locales = 'datos_historicos_hospital.csv'

# Inicialización estricta de estados de sesión
if 'modelo_entrenado' not in st.session_state:
    st.session_state['modelo_entrenado'] = False
if 'umbral_jaen' not in st.session_state:
    st.session_state['umbral_jaen'] = 0.522
if 'X_jaen' not in st.session_state:
    st.session_state['X_jaen'] = None
if 'y_jaen' not in st.session_state:
    st.session_state['y_jaen'] = None
if 'columnas_jaen' not in st.session_state:
    st.session_state['columnas_jaen'] = ['Glucosa', 'Trigliceridos', 'Colesterol', 'Gamma_GT', 'Albumina', 'MCH', 'Magnesio', 'Hb_libre', 'PCR', 'proBNP']

# Carga de la persistencia del modelo entrenado
if os.path.exists(ruta_modelo):
    with open(ruta_modelo, 'rb') as archivo:
        datos = pickle.load(archivo)
        st.session_state['modelo_entrenado'] = True
        st.session_state['clf_jaen'] = datos['clf']
        st.session_state['imputer_jaen'] = datos['imputer']
        st.session_state['scaler_jaen'] = datos['scaler']
        st.session_state['columnas_jaen'] = datos['columnas']
        st.session_state['umbral_jaen'] = datos.get('umbral', 0.522)

# Función de limpieza de caracteres analíticos
def limpiar_valor_para_entrenamiento(val):
    if pd.isna(val): return np.nan
    val_str = str(val).strip().upper().replace(',', '.')
    if val_str in ['NP', 'MHC', 'MNR', '-', 'MAR', '']: return np.nan
    if '<' in val_str or '>' in val_str: return float(re.sub(r'[<>]', '', val_str).strip())
    try: return float(val_str)
    except: return np.nan

# --- 2. ESTILOS VISUALES INSTITUCIONALES (VERDE ANDALUZ Y OLIVA) ---
# Hemos seleccionado los códigos hexadecimales exactos de la Junta de Andalucía
st.markdown("""
<style>
    /* Fondo general */
    .stApp { background-color: #f8fafc; }
    .block-container { max-width: 1100px !important; padding-top: 2rem !important; }
    
    /* Logos y Cabeceras */
    div[data-testid="stImage"] { display: flex; align-items: center; justify-content: center; }
    div[data-testid="stImage"] img { 
        max-height: 90px !important; 
        width: auto !important; 
        border-radius: 0px !important; 
        box-shadow: none !important;
    }
    /* Verde Título Principal (Corporate Green - SAS) */
    h1 { color: #008f4c !important; font-family: 'Segoe UI', system-ui, sans-serif; font-weight: 800 !important; letter-spacing: -0.5px; }
    
    /* Gris Títulos Secundarios */
    h2, h3, h4 { color: #334155 !important; font-family: 'Segoe UI', system-ui, sans-serif; font-weight: 700 !important; }
    
    /* Configuración de Pestañas (Verde SAS) */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #e2e8f0; gap: 20px; }
    .stTabs [aria-selected="true"] { color: #008f4c !important; border-bottom: 3px solid #008f4c !important; font-weight: 700 !important; }
    
    /* Botones de Acción Clínicos (Verde Oliva Intenso) */
    .stButton>button { 
        background-color: #3b5a2f !important; 
        color: white !important; 
        font-weight: 600 !important; 
        border-radius: 6px !important;
        padding: 0.6rem 1.2rem !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        background-color: #2b4323 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-1px);
    }
    
    /* Métricas Estadísticas (Verde Oliva Intenso) */
    div[data-testid="stMetricValue"] { color: #3b5a2f; font-weight: 800; font-size: 2.2rem;}
    .streamlit-expanderHeader { font-weight: 600 !important; color: #475569 !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. CABECERA INSTITUCIONAL ---
header_col1, header_col2 = st.columns([1.2, 6], vertical_alignment="center")
with header_col1:
    if os.path.exists("huj.png"): st.image("huj.png", use_container_width=True) 
with header_col2:
    st.title("PROTOCOLO CARDIOGEN JAÉN")
    st.markdown("<p style='color: #64748b; font-size: 1.15rem; margin-top: -10px; font-weight: 500;'>PLATAFORMA ALGORÍTMICA DE CRIBADO INSTITUCIONAL</p>", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["EVALUACIÓN CLÍNICA", "CALIBRACIÓN DEL MOTOR", "AUDITORÍA E INFORMES"])

# ==========================================
# PESTAÑA 1: EVALUACIÓN EN CONSULTA
# ==========================================
with tab1:
    st.markdown("### 1. EXTRACCIÓN DE DATOS CLÍNICOS (LABORATORIO)")
    pdf_subido = st.file_uploader("Subir analítica del SAS (Formato PDF)", type=["pdf"], label_visibility="collapsed")
    valores_extraidos = {k: 0.0 for k in st.session_state['columnas_jaen']}
    
    if pdf_subido is not None:
        try:
            texto_completo = ""
            with pdfplumber.open(pdf_subido) as pdf:
                for pagina in pdf.pages:
                    t = pagina.extract_text(layout=True)
                    if t: texto_completo += t + "\n"
            
            if not texto_completo.strip():
                st.warning("ATENCIÓN: El documento parece ser una imagen escaneada o no contiene texto digital. El sistema de extracción óptica requiere un PDF nativo del SAS. Por favor, introduzca los valores manualmente.")
            else:
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
                st.success("Extracción digital completada con éxito.")
        except Exception as e: st.error(f"Error procesando documento PDF: {e}")

    with st.expander("REVISIÓN MANUAL DE VALORES EXTRAÍDOS", expanded=True):
        st.info("Nota del sistema: Los valores en '0.0' indican ausencia de dato en la analítica primaria. El modelo aplicará imputación estadística automatizada (máximo 2 variables permitidas).")
        cols = st.columns(2)
        for i, (k, v) in enumerate(valores_extraidos.items()):
            valores_extraidos[k] = cols[i % 2].number_input(k, value=float(v))
    
    st.write("")
    st.markdown("### 2. MOTOR PREDICTIVO MULTIVARIANTE")
    
    if st.button("EJECUTAR PROTOCOLO DE ESTRATIFICACIÓN", use_container_width=True):
        if not st.session_state['modelo_entrenado']:
            st.error("Error: El Score Institucional no está configurado. Por favor, realice la calibración inicial de la matriz en la pestaña correspondiente.")
        else:
            parametros_ausentes = sum(1 for v in valores_extraidos.values() if v == 0.0)
            
            if parametros_ausentes > 2:
                st.error(f"BLOQUEO DE SEGURIDAD CLÍNICA: Se han detectado {parametros_ausentes} parámetros ausentes. El protocolo exige un mínimo de 8 biomarcadores válidos sobre 10 para garantizar la precisión diagnóstica. Por favor, solicite una analítica más completa o revise la extracción manual.")
            else:
                datos_paciente = []
                for col in st.session_state['columnas_jaen']:
                    val = valores_extraidos.get(col, 0.0)
                    datos_paciente.append(np.nan if val == 0.0 else val)
                
                X_paciente = np.array(datos_paciente).reshape(1, -1)
                X_imputed = st.session_state['imputer_jaen'].transform(X_paciente)
                X_scaled = st.session_state['scaler_jaen'].transform(X_imputed)
                
                probabilidad = st.session_state['clf_jaen'].predict_proba(X_scaled)[0][1]
                umbral_clinico = st.session_state['umbral_jaen']
                
                st.markdown("---")
                col_res1, col_res2 = st.columns([1, 2])
                col_res1.metric("Probabilidad Predictiva", f"{probabilidad:.1%}")
                
                with col_res2:
                    if probabilidad >= umbral_clinico:
                        st.error(f"ATENCIÓN CLÍNICA: RIESGO SIGNIFICATIVO. El perfil bioquímico supera el umbral de seguridad local fijado en {umbral_clinico:.1%}. Se recomienda derivación para valoración específica de Amiloidosis.")
                        riesgo_texto = "ALTO RIESGO"
                    else:
                        st.success(f"VALORACIÓN: BAJO RIESGO CLÍNICO. La firma predictiva se mantiene por debajo del umbral de alarma institucional ({umbral_clinico:.1%}).")
                        riesgo_texto = "BAJO RIESGO"
                
                st.write("")
                with st.expander("GENERAR INFORME PARA HISTORIA CLÍNICA (DIRAYA)"):
                    informe = (
                        "========================================================\n"
                        "INFORME DE ESTRATIFICACIÓN - PROTOCOLO CARDIOGEN JAÉN\n"
                        "========================================================\n\n"
                        "RESULTADO DEL ANÁLISIS MULTIVARIANTE (MACHINE LEARNING):\n"
                        f"- Probabilidad predictiva del algoritmo: {probabilidad:.1%}\n"
                        f"- Punto de corte de seguridad (institucional): {umbral_clinico:.1%}\n"
                        f"- CATEGORIZACIÓN DE RIESGO CLÍNICO: {riesgo_texto}\n\n"
                        "PERFIL DE BIOMARCADORES DESTACADOS:\n"
                        f"- NT-proBNP: {valores_extraidos['proBNP']} pg/mL\n"
                        f"- Albúmina sérica: {valores_extraidos['Albumina']} g/dL\n"
                        f"- Magnesio sérico: {valores_extraidos['Magnesio']} mg/dL\n\n"
                        "* NOTA TÉCNICA: La probabilidad ha sido calculada evaluando\n"
                        "la firma bioquímica completa de 10 parámetros de rutina.\n"
                        f"Se ha aplicado imputación estadística automatizada sobre {parametros_ausentes} valores\n"
                        "ausentes en la analítica primaria (límite de seguridad: 2).\n"
                        "========================================================"
                    )
                    st.code(informe, language="text")

# ==========================================
# AUXILIAR: LOGICA DE PERSISTENCIA DE COHORTE
# ==========================================
def procesar_y_guardar_dataframe(df):
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
        st.session_state['X_jaen'] = df_limpio[columnas_finales].apply(lambda col: col.map(limpiar_valor_para_entrenamiento))
        st.session_state['y_jaen'] = df_limpio['Diagnóstico final']
        st.session_state['columnas_jaen'] = columnas_finales
    except Exception as e:
        st.error(f"Error procesando las columnas de la base de datos. Detalle: {e}")

# Carga automática de fondo si existe copia en disco
if os.path.exists(ruta_datos_locales) and st.session_state['X_jaen'] is None:
    try:
        df_local = pd.read_csv(ruta_datos_locales)
        procesar_y_guardar_dataframe(df_local)
    except:
        pass

# ==========================================
# PESTAÑA 2: CALIBRACIÓN DEL MOTOR
# ==========================================
with tab2:
    st.markdown("### CALIBRACIÓN DEL MOTOR PREDICTIVO")
    st.info("Módulo restringido para actualización de la matriz de pesos del modelo local mediante Regresión Logística Multivariante.")
    
    st.markdown("#### Carga de archivos de cohorte")
    archivo_csv = st.file_uploader("Cargar Base de Datos Histórica (.xlsx / .csv)", type=["xlsx", "csv"], label_visibility="collapsed")
    
    if archivo_csv:
        df_subido = pd.read_csv(archivo_csv) if archivo_csv.name.endswith('.csv') else pd.read_excel(archivo_csv)
        df_subido.to_csv(ruta_datos_locales, index=False)
        procesar_y_guardar_dataframe(df_subido)
        st.toast("Base de datos almacenada correctamente en el sistema local.")
    elif st.session_state['X_jaen'] is not None:
        st.caption("Información del sistema: Se ha detectado una cohorte histórica guardada en disco. No es necesario volver a subir el archivo.")

    if st.session_state['X_jaen'] is not None:
        if st.button("INICIAR CALIBRACIÓN INSTITUCIONAL", use_container_width=True):
            imputer = SimpleImputer(strategy='median')
            X_imp = imputer.fit_transform(st.session_state['X_jaen'])
            scaler = StandardScaler()
            X_sca = scaler.fit_transform(X_imp)
            
            clf = LogisticRegression(max_iter=2000, class_weight='balanced')
            clf.fit(X_sca, st.session_state['y_jaen'])
            
            umbral_actual = st.session_state.get('umbral_jaen', 0.522)
            
            with open(ruta_modelo, 'wb') as archivo:
                pickle.dump({'clf': clf, 'imputer': imputer, 'scaler': scaler, 'columnas': st.session_state['columnas_jaen'], 'umbral': umbral_actual}, archivo)
                
            df_coef = pd.DataFrame({
                'Biomarcador': st.session_state['columnas_jaen'], 
                'Coeficiente (Peso matemático)': np.round(clf.coef_[0], 4)
            }).sort_values(by='Coeficiente (Peso matemático)', ascending=False)
            
            st.session_state['df_coef_jaen'] = df_coef
            st.session_state['modelo_entrenado'] = True
            st.session_state['clf_jaen'] = clf
            st.session_state['imputer_jaen'] = imputer
            st.session_state['scaler_jaen'] = scaler
            
            st.success("Operación completada: Motor predictivo calibrado y matriz de pesos persistida.")

    # Renderizado Vertical Óptimo
    if 'df_coef_jaen' in st.session_state:
        st.write("")
        st.divider()
        
        st.markdown("#### TABLA DE COEFICIENTES")
        st.markdown("<p style='color: #64748b; font-size: 0.9rem; margin-top: -10px;'>Pesos numéricos exactos calculados para la ecuación Logit de Jaén.</p>", unsafe_allow_html=True)
        st.dataframe(st.session_state['df_coef_jaen'], use_container_width=True, hide_index=True)
        
        st.write("")
        st.markdown("#### IMPACTO PARAMÉTRICO EN EL RIESGO CLÍNICO")
        st.markdown("<p style='color: #64748b; font-size: 0.9rem; margin-top: -10px;'>En verde se representan los factores de riesgo (suman al score); en rojo los factores protectores (restan).</p>", unsafe_allow_html=True)
        
        # COLOR ACTUALIZADO: Verde Oliva SAS vs Rojo Fisiológico
        grafico = alt.Chart(st.session_state['df_coef_jaen']).mark_bar().encode(
            x=alt.X('Coeficiente (Peso matemático):Q', title='Peso Predictivo (Regresión Logística)'),
            y=alt.Y('Biomarcador:N', sort='x', title=''), 
            color=alt.condition(alt.datum['Coeficiente (Peso matemático)'] > 0, alt.value('#3b5a2f'), alt.value('#b91c1c')),
            tooltip=['Biomarcador', 'Coeficiente (Peso matemático)']
        ).properties(height=400).interactive()
        
        st.altair_chart(grafico, use_container_width=True)

# ==========================================
# PESTAÑA 3: AUDITORÍA E INFORMES (PERSISTENTE Y COLORES INSTITUCIONALES)
# ==========================================
with tab3:
    st.markdown("### AUDITORÍA CLÍNICA Y OPTIMIZACIÓN DE UMBRAL")
    st.info("Módulo de validación interna (Hold-out 75/25). Calcula el Área Bajo la Curva (AUC) y determina la efectividad clínica mediante métricas de rendimiento basadas en el umbral local.")
    
    if st.session_state['X_jaen'] is not None:
        if st.button("GENERAR INFORME DE AUDITORÍA Y ACTUALIZAR UMBRAL", use_container_width=True):
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state['X_jaen'], st.session_state['y_jaen'], 
                test_size=0.25, random_state=42, stratify=st.session_state['y_jaen']
            )
            
            imp_val = SimpleImputer(strategy='median')
            sca_val = StandardScaler()
            X_train_sca = sca_val.fit_transform(imp_val.fit_transform(X_train))
            X_test_sca = sca_val.transform(imp_val.transform(X_test))
            
            # Evaluación del modelo lineal de regresión
            clf_lr = LogisticRegression(max_iter=2000, class_weight='balanced')
            clf_lr.fit(X_train_sca, y_train)
            y_pred_prob_lr = clf_lr.predict_proba(X_test_sca)[:, 1]
            fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_pred_prob_lr)
            auc_lr = auc(fpr_lr, tpr_lr)
            
            # Optimización por Youden
            youden_idx = np.argmax(tpr_lr - fpr_lr)
            nuevo_umbral = thresholds_lr[youden_idx]
            st.session_state['umbral_jaen'] = nuevo_umbral
            
            if st.session_state['modelo_entrenado']:
                with open(ruta_modelo, 'rb') as archivo: d = pickle.load(archivo)
                d['umbral'] = nuevo_umbral
                with open(ruta_modelo, 'wb') as archivo: pickle.dump(d, archivo)
            
            # Evaluación de la arquitectura avanzada XGBoost
            scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
            clf_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)
            clf_xgb.fit(X_train_sca, y_train)
            y_pred_prob_xgb = clf_xgb.predict_proba(X_test_sca)[:, 1]
            fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob_xgb)
            auc_xgb = auc(fpr_xgb, tpr_xgb)
            
            # Procesamiento de la Matriz de Confusión diagnóstica
            y_pred_binario = (y_pred_prob_lr >= nuevo_umbral).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binario).ravel()
            
            sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
            especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
            vpp = tp / (tp + fp) if (tp + fp) > 0 else 0
            vpn = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Datos ROC unificados
            df_lr = pd.DataFrame({'FPR': fpr_lr, 'TPR': tpr_lr, 'Modelo': f'Regresion Lineal (AUC = {auc_lr:.3f})'})
            df_xgb = pd.DataFrame({'FPR': fpr_xgb, 'TPR': tpr_xgb, 'Modelo': f'XGBoost Avanzado (AUC = {auc_xgb:.3f})'})
            df_ref = pd.DataFrame({'FPR': [0, 1], 'TPR': [0, 1], 'Modelo': 'Referencia Aleatoria'})
            df_roc = pd.concat([df_lr, df_xgb, df_ref])
            
            # Volcado al estado de sesión
            st.session_state['auditoria_lista'] = True
            st.session_state['nuevo_umbral_calculado'] = nuevo_umbral
            st.session_state['m_sensibilidad'] = sensibilidad
            st.session_state['m_especificidad'] = especificidad
            st.session_state['m_vpp'] = vpp
            st.session_state['m_vpn'] = vpn
            st.session_state['auc_lr'] = auc_lr
            st.session_state['auc_xgb'] = auc_xgb
            st.session_state['df_roc_data'] = df_roc

    # Bloque de visualización permanente de la auditoría
    if 'auditoria_lista' in st.session_state:
        st.write("")
        st.divider()
        st.success(f"Punto de Corte Institucional fijado matemáticamente por Youden en: {st.session_state['nuevo_umbral_calculado']*100:.1f}%")
        
        st.markdown("#### MATRIZ DE RENDIMIENTO DIAGNÓSTICO (Umbral Optimizado)")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Sensibilidad (Detección)", f"{st.session_state['m_sensibilidad']:.1%}", help="Capacidad del sistema para capturar enfermos reales.")
        m_col2.metric("Especificidad (Discriminación)", f"{st.session_state['m_especificidad']:.1%}", help="Capacidad del sistema para descartar falsos positivos.")
        m_col3.metric("Valor Predictivo Positivo (VPP)", f"{st.session_state['m_vpp']:.1%}", help="Probabilidad de estar enfermo ante una alerta positiva.")
        m_col4.metric("Valor Predictivo Negativo (VPN)", f"{st.session_state['m_vpn']:.1%}", help="Seguridad clínica ante una alerta de bajo riesgo.")
        
        st.markdown("---")
        st.markdown("#### CURVAS ROC COMPARATIVAS DE MODELOS")
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Poder Predictivo Lineal (AUC)", f"{st.session_state['auc_lr']:.3f}")
        col_m2.metric("Poder Predictivo XGBoost (AUC)", f"{st.session_state['auc_xgb']:.3f}")
        
        # COLOR ACTUALIZADO EN ROC: Verde Oliva (Lineal), Verde SAS (XGBoost) y Slate (Referencia)
        roc_chart = alt.Chart(st.session_state['df_roc_data']).mark_line(size=3).encode(
            x=alt.X('FPR:Q', title='Tasa de Falsos Positivos (1 - Especificidad)'),
            y=alt.Y('TPR:Q', title='Tasa de Verdaderos Positivos (Sensibilidad)'),
            color=alt.Color('Modelo:N', scale=alt.Scale(
                domain=[f"Regresion Lineal (AUC = {st.session_state['auc_lr']:.3f})", f"XGBoost Avanzado (AUC = {st.session_state['auc_xgb']:.3f})", 'Referencia Aleatoria'],
                range=['#3b5a2f', '#008f4c', '#94a3b8']
            ), legend=alt.Legend(title="Algoritmo Predictivo", orient='bottom-right')),
            strokeDash=alt.condition(alt.datum.Modelo == 'Referencia Aleatoria', alt.value([5, 5]), alt.value([0])),
            tooltip=['Modelo', 'FPR', 'TPR']
        ).properties(height=480).interactive()
        
        st.altair_chart(roc_chart, use_container_width=True)
    else:
        if st.session_state['X_jaen'] is None:
            st.warning("Por favor, cargue el archivo histórico de calibración en la pestaña 2 para habilitar las opciones de auditoría de la cohorte.")
