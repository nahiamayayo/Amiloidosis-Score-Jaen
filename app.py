import streamlit as st
import pandas as pd
from motor_algoritmo import procesar_analitica_paciente, calcular_score_bruto, evaluar_perfil_riesgo
            
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
                
                # Mostrar métricas en columnas
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
                
                # Desglosar qué variables saltaron
                if alertas:
                    st.markdown("**Marcadores detectados en rango de infiltración/congestión:**")
                    for alerta in alertas:
                        st.markdown(f"- {alerta}")

st.set_page_config(page_title="ATTR-Lab Centinela", page_icon="🫀", layout="wide")

st.title("🫀 Proyecto Centinela: ATTR-Lab (Hospital de Jaén)")
st.markdown("Herramienta de cribado inteligente para Amiloidosis Cardíaca basada en parámetros de laboratorio de rutina.")

tab1, tab2 = st.tabs(["Calculadora Individual", "Validación por Lotes (Archivo)"])

with tab1:
    st.header("Evaluación de Paciente Individual")
    st.markdown("Introduce los valores analíticos del paciente. Deja en blanco los campos no disponibles.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Perfil Metabólico")
        glucosa = st.number_input("Glucosa (mg/dL)", value=None)
        trigliceridos = st.number_input("Triglicéridos (mg/dL)", value=None)
        colesterol = st.number_input("Colesterol Total (mg/dL)", value=None)

    with col2:
        st.subheader("Perfil Hepático")
        colinesterasa = st.number_input("Colinesterasa (kU/L)", value=None)
        gamma_gt = st.number_input("Gamma-GT (U/L)", value=None)
        albumina = st.number_input("Albúmina (g/L)", value=None)

    with col3:
        st.subheader("Perfil Hemático / Otros")
        mch = st.number_input("MCH (pg)", value=None)
        cloruro = st.number_input("Cloruro (mmol/L)", value=None)
        magnesio = st.number_input("Magnesio (mmol/L)", value=None)
        hemoglobina_libre = st.number_input("Hb libre (µmol/L)", value=None)
        alfa_amilasa = st.number_input("Alfa-amilasa (U/L)", value=None)
        pcr = st.number_input("PCR (mg/dL)", value=None)

        if st.button("Calcular Riesgo (Provisional)", type="primary"):
            datos_paciente = {
                'Trigliceridos': trigliceridos, 'Glucosa': glucosa, 'Colinesterasa': colinesterasa,
                'Cloruro': cloruro, 'Albumina': albumina, 'Alfa_amilasa': alfa_amilasa,
                'PCR': pcr, 'Hemoglobina_libre': hemoglobina_libre, 'Magnesio': magnesio,
                'Gamma_GT': gamma_gt, 'MCH': mch, 'Colesterol': colesterol
            }

        resultado = procesar_analitica_paciente(datos_paciente)

        if resultado['estado'] == "ERROR":
            st.error(resultado['mensaje'])
        else:
            st.success(resultado['mensaje'])
            st.info("Nota: El cálculo del score definitivo se activará tras la validación de la cohorte local.")

with tab2:
    st.header("Validación Retrospectiva de Cohorte")
    st.markdown("Sube el archivo Excel o CSV anonimizado.")
    archivo_subido = st.file_uploader("Selecciona un archivo (.xlsx, .csv)", type=["xlsx", "csv"])

    if archivo_subido is not None:
        st.success("Archivo cargado correctamente. Listo para procesar.")
        if st.button("Procesar Cohorte"):
            st.warning("Función en desarrollo.")
