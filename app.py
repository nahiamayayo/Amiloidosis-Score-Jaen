import streamlit as st
import pandas as pd
import re
import pdfplumber
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo

st.set_page_config(page_title="Amiloidosis-Score-Jaén", page_icon="🫀", layout="wide")

st.title("🫀 Amiloidosis-Score-Jaén")
st.markdown("Herramienta de cribado inteligente para Amiloidosis Cardíaca basada en parámetros de rutina.")

tab1, tab2 = st.tabs(["Calculadora Individual", "Validación por Lotes (Archivo)"])

with tab1:
    st.header("Evaluación de Paciente Individual")
    
    st.info("💡 Novedad: Sube el PDF de la analítica anonimizada y el sistema extraerá los valores automáticamente.")
    pdf_subido = st.file_uploader("Subir Analítica (PDF)", type=["pdf"])
    
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
                
            # Diccionario con tuplas de patrones: (1. Nombre primero, 2. Número primero)
            patrones = {
                'Glucosa': (r'\bGlucosa\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\bGlucosa\b'),
                'Trigliceridos': (r'\bTriglic[eé]ridos\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\bTriglic[eé]ridos\b'),
                'Colesterol': (r'\bColesterol(?: total)?\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\bColesterol(?: total)?\b'),
                'Colinesterasa': (r'\bColinesterasa\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\bColinesterasa\b'),
                'Gamma_GT': (r'(?:Gamma[\s-]*GT|Gamma[\s-]*glutamil[\s-]*transferasa)\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?(?:Gamma[\s-]*GT|Gamma[\s-]*glutamil[\s-]*transferasa)'),
                'Albumina': (r'\bAlb[uú]mina\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\bAlb[uú]mina\b'),
                'MCH': (r'\b(?:MCH|Hemoglobina corpuscular media)\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\b(?:MCH|Hemoglobina corpuscular media)\b'),
                'Cloruro': (r'\bCloruro\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\bCloruro\b'),
                'Magnesio': (r'\bMagnesio\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\bMagnesio\b'),
                'Hb_libre': (r'\b(?!Hemoglobina corpuscular\b)Hemoglobina\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\b(?!Hemoglobina corpuscular\b)Hemoglobina\b'),
                'Alfa_amilasa': (r'\b(?:Alfa[- ]amilasa|amilasa)\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\b(?:Alfa[- ]amilasa|amilasa)\b'),
                'PCR': (r'\b(?:PCR|Prote[ií]na C reactiva)\b\D*?(\d+[.,]\d+|\d+)', r'(\d+[.,]\d+|\d+)\D*?\b(?:PCR|Prote[ií]na C reactiva)\b')
            }

            variables_encontradas = 0
            for clave, (patron_despues, patron_antes) in patrones.items():
                # Estrategia A: Buscar si el número está a la derecha del nombre
                match = re.search(patron_despues, texto_completo, re.IGNORECASE)
                if match:
                    valor_str = match.group(1).replace(',', '.')
                    valores_extraidos[clave] = float(valor_str)
                    variables_encontradas += 1
                else:
                    # Estrategia B: Buscar si el número está a la izquierda (inversión de tabla)
                    match = re.search(patron_antes, texto_completo, re.IGNORECASE)
                    if match:
                        valor_str = match.group(1).replace(',', '.')
                        valores_extraidos[clave] = float(valor_str)
                        variables_encontradas += 1
            
            st.success(f"📄 Analítica procesada: Se han extraído {variables_encontradas} parámetros automáticamente.")
            
        except Exception as e:
            st.error(f"Error al leer el PDF: {e}")

    st.divider()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Perfil Metabólico")
        glucosa = st.number_input("Glucosa (mg/dL)", value=valores_extraidos['Glucosa'])
        trigliceridos = st.number_input("Triglicéridos (mg/dL)", value=valores_extraidos['Trigliceridos'])
        colesterol = st.number_input("Colesterol Total (mg/dL)", value=valores_extraidos['Colesterol'])
        
    with col2:
        st.subheader("Perfil Hepático")
        colinesterasa = st.number_input("Colinesterasa (kU/L)", value=valores_extraidos['Colinesterasa'])
        gamma_gt = st.number_input("Gamma-GT (U/L)", value=valores_extraidos['Gamma_GT'])
        albumina = st.number_input("Albúmina (g/L)", value=valores_extraidos['Albumina'])
        
    with col3:
        st.subheader("Perfil Hemático / Otros")
        mch = st.number_input("MCH (pg)", value=valores_extraidos['MCH'])
        cloruro = st.number_input("Cloruro (mmol/L)", value=valores_extraidos['Cloruro'])
        magnesio = st.number_input("Magnesio (mmol/L)", value=valores_extraidos['Magnesio'])
        hemoglobina_libre = st.number_input("Hb libre (µmol/L)", value=valores_extraidos['Hb_libre'])
        alfa_amilasa = st.number_input("Alfa-amilasa (U/L)", value=valores_extraidos['Alfa_amilasa'])
        pcr = st.number_input("PCR (mg/dL)", value=valores_extraidos['PCR'])

    if st.button("Calcular Riesgo (Provisional)", type="primary"):
        datos_paciente = {
            'Trigliceridos': trigliceridos, 'Glucosa': glucosa, 'Colinesterasa': colinesterasa,
            'Cloruro': cloruro, 'Albumina': albumina, 'Alfa_amilasa': alfa_amilasa,
            'PCR': pcr, 'Hemoglobina_libre': hemoglobina_libre, 'Magnesio': magnesio,
            'Gamma_GT': gamma_gt, 'MCH': mch, 'Colesterol': colesterol
        }
        
        resultado = procesar_analitica_paciente(datos_paciente)
        
        if resultado['estado'] == "ERROR":
            st.error(f"🛑 {resultado['mensaje']}")
        else:
            st.success("✅ " + resultado['mensaje'])
            
            datos_limpios = resultado['datos_procesados']
            score = calcular_score_bruto(datos_limpios)
            alertas = evaluar_perfil_riesgo(datos_limpios)
            
            st.divider()
            st.subheader("Resultados del Análisis Clínico")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric(label="Score Bruto (Logístico)", value=score)
                st.caption("Nota: Este número representa la suma ponderada temporal hasta validar la cohorte local.")
            
            with col_res2:
                if len(alertas) >= 3:
                    st.error(f"🚨 ALTA SOSPECHA: {len(alertas)} de 5 marcadores en rango de riesgo.")
                elif len(alertas) > 0:
                    st.warning(f"⚠️ RIESGO MODERADO/MIXTO: {len(alertas)} de 5 marcadores en rango de riesgo.")
                else:
                    st.success("🟢 BAJO RIESGO: Patrón bioquímico inconsistente con amiloidosis.")
            
            if alertas:
                st.markdown("**Marcadores detectados en rango de infiltración/congestión:**")
                for alerta in alertas:
                    st.markdown(f"- {alerta}")

with tab2:
    st.header("Validación Retrospectiva de Cohorte")
    st.markdown("Sube el archivo Excel o CSV anonimizado.")
    archivo_subido = st.file_uploader("Selecciona un archivo (.xlsx, .csv)", type=["xlsx", "csv"])
    
    if archivo_subido is not None:
        st.success("Archivo cargado correctamente. Listo para procesar.")
        if st.button("Procesar Cohorte"):
            st.warning("Función en desarrollo.")
