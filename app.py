import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
import os
import pickle
import altair as alt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
st.set_page_config(page_title="ATTR Lab", layout="wide", initial_sidebar_state="collapsed")

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
        # Soporte para la versión antigua (clf) y la nueva (clf_lr si quedó guardada)
        st.session_state['clf_jaen'] = datos.get('clf', datos.get('clf_lr'))
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
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .block-container { max-width: 1100px !important; padding-top: 2rem !important; }
    div[data-testid="stImage"] { display: flex; align-items: center; justify-content: center; }
    div[data-testid="stImage"] img { max-height: 90px !important; width: auto !important; border-radius: 0px !important; box-shadow: none !important; }
    h1 { color: #008f4c !important; font-family: 'Segoe UI', system-ui, sans-serif; font-weight: 800 !important; letter-spacing: -0.5px; }
    h2, h3, h4 { color: #334155 !important; font-family: 'Segoe UI', system-ui, sans-serif; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #e2e8f0; gap: 20px; }
    .stTabs [aria-selected="true"] { color: #008f4c !important; border-bottom: 3px solid #008f4c !important; font-weight: 700 !important; }
    .stButton>button { background-color: #3b5a2f !important; color: white !important; font-weight: 600 !important; border-radius: 6px !important; padding: 0.6rem 1.2rem !important; border: none !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: all 0.3s ease; text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { background-color: #2b4323 !important; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transform: translateY(-1px); }
    div[data-testid="stMetricValue"] { color: #3b5a2f; font-weight: 800; font-size: 2.2rem;}
    .streamlit-expanderHeader { font-weight: 600 !important; color: #475569 !important; }
    .instrucciones-caja { background-color: #ffffff; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #008f4c; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# --- 3. CABECERA INSTITUCIONAL ---
header_col1, header_col2 = st.columns([1.2, 6], vertical_alignment="center")
with header_col1:
    if os.path.exists("huj.png"): st.image("huj.png", use_container_width=True) 
with header_col2:
    st.title("ATTR LAB")
    st.markdown("<p style='color: #64748b; font-size: 1.15rem; margin-top: -10px; font-weight: 500;'>HERRAMIENTA DE LABORATORIO PARA LA ESTRATIFICACIÓN DE RIESGO DE AMILOIDOSIS CARDÍACA POR TRANSTIRETINA</p>", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["EVALUACIÓN CLÍNICA", "CALIBRACIÓN DEL MOTOR", "AUDITORÍA E INFORMES"])

# ==========================================
# PESTAÑA 1: EVALUACIÓN EN CONSULTA
# ==========================================
with tab1:
    st.markdown("### IMPORTACIÓN DE ANALÍTICA")
    
    st.markdown("""
    <div class="instrucciones-caja">
        <h4 style="margin-top: 0; color: #334155; margin-bottom: 15px;">Instrucciones de uso:</h4>
        <div style="display: flex; gap: 15px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 200px; background-color: #f1f5f9; padding: 12px; border-radius: 6px; border-left: 3px solid #008f4c;">
                <strong>1. Preparación</strong><br>
                <span style="color: #475569; font-size: 0.9em;">Descargue el PDF original del SAS y asegúrese de que la muestra esté <b>anonimizada</b>.</span>
            </div>
            <div style="flex: 1; min-width: 200px; background-color: #f1f5f9; padding: 12px; border-radius: 6px; border-left: 3px solid #008f4c;">
                <strong>2. Carga</strong><br>
                <span style="color: #475569; font-size: 0.9em;">Arrastre el archivo o haga clic en la zona de subida situada justo debajo.</span>
            </div>
            <div style="flex: 1; min-width: 200px; background-color: #f1f5f9; padding: 12px; border-radius: 6px; border-left: 3px solid #008f4c;">
                <strong>3. Extracción</strong><br>
                <span style="color: #475569; font-size: 0.9em;">El lector óptico procesará los 10 biomarcadores clave de forma automática.</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    pdf_subido = st.file_uploader("Arrastre el archivo PDF o haga clic para subir", type=["pdf"], label_visibility="collapsed")
        
    valores_extraidos = {k: 0.0 for k in st.session_state['columnas_jaen']}
    
    if pdf_subido is not None:
        try:
            texto_completo = ""
            with pdfplumber.open(pdf_subido) as pdf:
                for pagina in pdf.pages:
                    t = pagina.extract_text(layout=True)
                    if t: texto_completo += t + "\n"
            
            if not texto_completo.strip():
                st.warning("ATENCIÓN: El documento parece ser una imagen escaneada o no contiene texto digital. El sistema de extracción óptica requiere un PDF nativo. Por favor, introduzca los valores manualmente.")
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

    st.write("")
    with st.expander("VERIFICACIÓN DE PARÁMETROS", expanded=True):
        st.info("**Nota de seguridad:** Revise que los valores extraídos son correctos. Los valores en '0.0' indican ausencia de dato en la analítica primaria. El sistema ajustará el cálculo usando la media del hospital (máximo 2 variables faltantes permitidas).")
        cols = st.columns(2)
        for i, (k, v) in enumerate(valores_extraidos.items()):
            valores_extraidos[k] = cols[i % 2].number_input(k, value=float(v))
    
    st.write("")
    st.markdown("### ESTRATIFICACIÓN DE RIESGO")
    
    if st.button("CALCULAR NIVEL DE RIESGO", use_container_width=True):
        if not st.session_state['modelo_entrenado'] or st.session_state.get('clf_jaen') is None:
            st.error("Error: El Score Institucional no está configurado. Por favor, realice la calibración inicial de la matriz en la pestaña 'CALIBRACIÓN DEL MOTOR'.")
        else:
            datos_paciente = []
            valores_faltantes = 0
            
            for col in st.session_state['columnas_jaen']:
                val = valores_extraidos.get(col, 0.0)
                if val == 0.0:
                    valores_faltantes += 1
                datos_paciente.append(np.nan if val == 0.0 else val)
            
            if valores_faltantes > 2:
                st.error(f"❌ ATENCIÓN CLÍNICA: Faltan {valores_faltantes} parámetros en la analítica. Para garantizar la seguridad del diagnóstico el sistema exige un máximo de 2 variables ausentes. Por favor, complete los datos manualmente en la sección superior.")
            else:
                X_paciente = np.array(datos_paciente).reshape(1, -1)
                X_imputed = st.session_state['imputer_jaen'].transform(X_paciente)
                X_scaled = st.session_state['scaler_jaen'].transform(X_imputed)
                
                # Ejecución de Predicción (Regresión Logística)
                probabilidad = st.session_state['clf_jaen'].predict_proba(X_scaled)[0][1]
                umbral_clinico = st.session_state['umbral_jaen']
                
                st.markdown("---")
                st.markdown("### RESULTADO DE ESTRATIFICACIÓN")
                
                col_res1, col_res2 = st.columns([1, 2])
                col_res1.metric("Probabilidad (Score Clínico)", f"{probabilidad:.1%}")
                
                with col_res2:
                    if probabilidad >= umbral_clinico:
                        st.error(f"ATENCIÓN CLÍNICA: RIESGO SIGNIFICATIVO. El perfil supera el umbral del {umbral_clinico*100:.1f}%. Se recomienda derivación.")
                        riesgo_texto = "ALTO RIESGO"
                    else:
                        st.success(f"VALORACIÓN: BAJO RIESGO CLÍNICO. El perfil se mantiene por debajo del umbral del {umbral_clinico*100:.1f}%.")
                        riesgo_texto = "BAJO RIESGO"
                
                st.write("")
                with st.expander("GENERAR INFORME DE ESTRATIFICACIÓN"):
                    informe = (
                        "========================================================\n"
                        "INFORME DE ESTRATIFICACIÓN - ATTR LAB\n"
                        "========================================================\n\n"
                        "RESULTADOS DEL ANÁLISIS (Regresión Logística Multivariante):\n"
                        f"- Probabilidad Predictiva: {probabilidad:.1%}\n"
                        f"- CATEGORIZACIÓN FINAL DE RIESGO: {riesgo_texto}\n\n"
                        
                        "* AVISO CLÍNICO LEGAL: Este informe es una herramienta de cribado orientativa.\n"
                        "No constituye un diagnóstico definitivo ni sustituye el juicio clínico.\n"
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
    
    st.warning("⚠️ **ZONA TÉCNICA AVANZADA:** Este módulo reconfigura la matriz matemática del modelo mediante Regresión Logística para todo el hospital. Se recomienda su uso exclusivo al equipo técnico o dirección del laboratorio.")
    desbloqueo_calibracion = st.checkbox("Entiendo las implicaciones. Desbloquear módulo de calibración.")
    
    if desbloqueo_calibracion:
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

                # Entrenar Regresión Logística
                clf = LogisticRegression(max_iter=2000, class_weight='balanced')
                clf.fit(X_sca, st.session_state['y_jaen'])
                
                umbral_actual = st.session_state.get('umbral_jaen', 0.522)

                # Persistencia
                with open(ruta_modelo, 'wb') as archivo:
                    pickle.dump({
                        'clf': clf, 
                        'imputer': imputer, 
                        'scaler': scaler, 
                        'columnas': st.session_state['columnas_jaen'], 
                        'umbral': umbral_actual
                    }, archivo)

                st.session_state['modelo_entrenado'] = True
                st.session_state['clf_jaen'] = clf
                st.session_state['imputer_jaen'] = imputer
                st.session_state['scaler_jaen'] = scaler

                st.success("Operación completada: Motor predictivo calibrado y guardado de forma persistente.")

    if st.session_state['modelo_entrenado'] and st.session_state.get('clf_jaen') is not None:
        st.write("")
        st.divider()
        st.markdown("### AUDITORÍA DE MATRIZ DE PESOS")
        st.markdown("<p style='color: #64748b; font-size: 0.95rem;'>Visualización de cómo el algoritmo lineal pondera fisiológicamente los biomarcadores.</p>", unsafe_allow_html=True)
        
        # Preparar DataFrames para la gráfica
        df_coef_lr = pd.DataFrame({
            'Biomarcador': st.session_state['columnas_jaen'],
            'Valor': np.round(st.session_state['clf_jaen'].coef_[0], 4)
        })

        with st.expander("📋 DETALLE TÉCNICO PARA LABORATORIO (Fórmula)", expanded=False):
            st.markdown("Pesos para implementar en el sistema LIS:")
            intercepto = st.session_state['clf_jaen'].intercept_[0]
            st.dataframe(df_coef_lr.rename(columns={'Valor': 'Peso (Coeficiente)'}), use_container_width=True, hide_index=True)
            st.write(f"**Intercepto (Beta 0):** {intercepto:.4f}")
            st.markdown("`z = Intercepto + (Peso1*Val1) + ... + (Peson*Valn)`")
            st.markdown("`Probabilidad = 1 / (1 + exp(-z))`")

        st.markdown("#### IMPACTO LINEAL (Regresión Logística)")
        grafico_lr = alt.Chart(df_coef_lr).mark_bar().encode(
            x=alt.X('Valor:Q', title='Peso Predictivo (Coeficiente)'),
            y=alt.Y('Biomarcador:N', sort='-x', title=''), 
            color=alt.condition(alt.datum['Valor'] > 0, alt.value('#008f4c'), alt.value('#64748b')),
            tooltip=['Biomarcador', 'Valor']
        ).properties(height=350)
        st.altair_chart(grafico_lr, use_container_width=True) 
        
        st.markdown("""
        <div style='font-size: 0.9rem; color: #475569; background-color: #f8fafc; padding: 15px; border-radius: 6px; border-left: 4px solid #008f4c; margin-bottom: 2rem;'>
            Interpretación Clínica: Este modelo aplica una ecuación matemática estricta. Las barras hacia la derecha (verdes) indican biomarcadores donde valores más altos se asocian a un mayor riesgo de amiloidosis. Las barras hacia la izquierda (grises) indican biomarcadores que exhiben una relación inversa: en estos casos, un valor en sangre anormalmente bajo es lo que alerta del riesgo de enfermedad.
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PESTAÑA 3: VALIDACIÓN Y RENDIMIENTO CLÍNICO
# ==========================================
with tab3:
    st.markdown("### VALIDACIÓN Y RENDIMIENTO CLÍNICO")
    
    st.markdown("""
    <div style="background-color: #f1f5f9; padding: 1.2rem; border-radius: 8px; border-left: 4px solid #3b5a2f; margin-bottom: 2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <h4 style="margin-top: 0; color: #334155; margin-bottom: 10px;">Auditoría Clínica y Calidad Predictiva</h4>
        <p style="color: #475569; font-size: 0.95rem; margin-bottom: 0;">
            Este módulo garantiza que el algoritmo no está simplemente "memorizando" historiales, sino aprendiendo a diagnosticar. 
            Para ello, el sistema se auto-examina aislando a un <b>25% de pacientes históricos de forma ciega</b>. 
            En base a este examen, recalibramos automáticamente el umbral de alerta (Punto óptimo) para <b>maximizar la detección de casos reales (sensibilidad) reduciendo al máximo las falsas alarmas (especificidad)</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['X_jaen'] is not None:
        if st.button("GENERAR INFORME DE VALIDACIÓN Y RECALIBRAR UMBRAL", use_container_width=True):
            X_train, X_test, y_train, y_test = train_test_split(
                st.session_state['X_jaen'], st.session_state['y_jaen'], 
                test_size=0.25, random_state=42, stratify=st.session_state['y_jaen']
            )
            
            imp_val = SimpleImputer(strategy='median')
            sca_val = StandardScaler()
            X_train_sca = sca_val.fit_transform(imp_val.fit_transform(X_train))
            X_test_sca = sca_val.transform(imp_val.transform(X_test))
            
            # Evaluación del modelo lineal
            clf_lr_val = LogisticRegression(max_iter=2000, class_weight='balanced')
            clf_lr_val.fit(X_train_sca, y_train)
            y_pred_prob_lr = clf_lr_val.predict_proba(X_test_sca)[:, 1]
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
            
            y_pred_binario = (y_pred_prob_lr >= nuevo_umbral).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binario).ravel()
            
            st.session_state.update({
                'auditoria_lista': True, 'nuevo_umbral_calculado': nuevo_umbral,
                'm_sensibilidad': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'm_especificidad': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'm_vpp': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'm_vpn': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'auc_lr': auc_lr,
                'df_roc_data': pd.DataFrame({'FPR': fpr_lr, 'TPR': tpr_lr})
            })
            st.rerun()

    if 'auditoria_lista' in st.session_state:
        st.success(f"Punto de corte óptimo (Youden): **{st.session_state['nuevo_umbral_calculado']*100:.1f}%**")
        
        st.markdown("#### MÉTRICAS DE EFECTIVIDAD CLÍNICA")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sensibilidad", f"{st.session_state['m_sensibilidad']:.1%}", help="Capacidad para detectar enfermos (evita falsos negativos).")
        c2.metric("Especificidad", f"{st.session_state['m_especificidad']:.1%}", help="Capacidad para descartar sanos (evita falsos positivos).")
        c3.metric("VPP", f"{st.session_state['m_vpp']:.1%}", help="Probabilidad de enfermedad tras una alerta positiva.")
        c4.metric("VPN", f"{st.session_state['m_vpn']:.1%}", help="Seguridad clínica ante un resultado bajo riesgo.")
        
        st.markdown("---")
        st.markdown("#### CURVA DE RENDIMIENTO (ROC)")
        
        # Corregimos los nombres para que el valor del AUC se imprima explícitamente en la leyenda
        texto_leyenda_modelo = f"Regresión Logística (AUC = {st.session_state['auc_lr']:.3f})"
        
        df_modelo = st.session_state['df_roc_data'].copy()
        df_modelo['Modelo'] = texto_leyenda_modelo
        
        df_ref = pd.DataFrame({
            'FPR': [0, 1], 
            'TPR': [0, 1], 
            'Modelo': 'Referencia Aleatoria (Gris)'
        })
        
        df_plot = pd.concat([df_modelo, df_ref])
        
        # Generación del gráfico con la leyenda corregida y limpia
        roc_chart = alt.Chart(df_plot).mark_line(size=3).encode(
            x=alt.X('FPR', title='Tasa de Falsos Positivos (1 - Especificidad)'),
            y=alt.Y('TPR', title='Tasa de Verdaderos Positivos (Sensibilidad)'),
            color=alt.Color('Modelo', scale=alt.Scale(
                domain=[texto_leyenda_modelo, 'Referencia Aleatoria (Gris)'],
                range=['#008f4c', '#94a3b8']
            ), title="Leyenda del Motor Predictivo"),
            strokeDash=alt.condition(
                alt.datum.Modelo == 'Referencia Aleatoria (Gris)', 
                alt.value([5, 5]), 
                alt.value([0])
            )
        ).properties(height=380).interactive()
        
        st.altair_chart(roc_chart, use_container_width=True)
