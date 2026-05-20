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
    st.info("💡 Sube el PDF de la analítica anonimizada y el sistema extraerá los valores automáticamente.")
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
            
            lineas = texto_completo.split('\n')
            
            # Diccionario blindado contra espacios extraños en PDFs (\s+ admite cualquier cantidad de espacios)
            patrones_nombres = {
                'Glucosa': r'Glucosa',
                'Trigliceridos': r'Triglic[eé]ridos|Triglic',
                'Colesterol': r'Colesterol',
                'Colinesterasa': r'Colinesterasa',
                'Gamma_GT': r'Gamma[\s-]*glutamil[\s-]*transferasa|Gamma[\s-]*GT|G\.G\.T',
                'Albumina': r'Alb[uú]mina',
                'MCH': r'Hemoglobina\s+corpuscular\s+media|HCM',
                'Cloruro': r'Cloruro',
                'Magnesio': r'Magnesio',
                'Hb_libre': r'Hemoglobina(?!\s*corpuscular|\s*glicosilada)',
                'Alfa_amilasa': r'Alfa[\s-]*amilasa|Amilasa',
                'PCR': r'Prote[ií]na\s+C\s+reactiva|PCR'
            }

            variables_encontradas = 0
            
            for i, linea in enumerate(lineas):
                for clave, patron in patrones_nombres.items():
                    if valores_extraidos[clave] is None:
                        match_nombre = re.search(patron, linea, re.IGNORECASE)
                        if match_nombre:
                            # Busca el primer número en lo que queda de la línea tras encontrar el nombre
                            linea_resto = linea[match_nombre.end():]
                            match_numero = re.search(r'(\d+[.,]\d+|\d+)', linea_resto)
                            
                            # Si la línea se ha cortado visualmente y no hay número, busca en la línea de abajo
                            if not match_numero and i + 1 < len(lineas):
                                match_numero = re.search(r'(\d+[.,]\d+|\d+)', lineas[i+1])
                                
                            if match_numero:
                                valor_str = match_numero.group(1).replace(',', '.')
                                valores_extraidos[clave] = float(valor_str)
                                variables_encontradas += 1
            
            st.success(f"📄 Analítica procesada: Se han extraído {variables_encontradas} parámetros automáticamente.")
        except Exception as e:
            st.error(f"Error al leer el PDF: {e}")

    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        glucosa = st.number_input("Glucosa (mg/dL)", value=valores_extraidos['Glucosa'])
        trigliceridos = st.number_input("Triglicéridos (mg/dL)", value=valores_extraidos['Trigliceridos'])
        colesterol = st.number_input("Colesterol Total (mg/dL)", value=valores_extraidos['Colesterol'])
    with col2:
        colinesterasa = st.number_input("Colinesterasa (kU/L)", value=valores_extraidos['Colinesterasa'])
        gamma_gt = st.number_input("Gamma-GT (U/L)", value=valores_extraidos['Gamma_GT'])
        albumina = st.number_input("Albúmina (g/L)", value=valores_extraidos['Albumina'])
    with col3:
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
            'PCR': pcr, 'Hb_libre': hemoglobina_libre, 'Magnesio': magnesio,
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
            with col_res2:
                if len(alertas) >= 3:
                    st.error(f"🚨 ALTA SOSPECHA: {len(alertas)} de 5 marcadores en riesgo.")
                elif len(alertas) > 0:
                    st.warning(f"⚠️ RIESGO MODERADO: {len(alertas)} marcadores en riesgo.")
                else:
                    st.success("🟢 BAJO RIESGO")
            
            if alertas:
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
