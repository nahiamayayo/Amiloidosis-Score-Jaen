import pandas as pd
import numpy as np

# Coeficientes de la Regresión Logística
PESOS = {
    'Trigliceridos': -0.65, 'Glucosa': -0.63, 'Colinesterasa': -0.48,
    'Cloruro': -0.38, 'Albumina': -0.22, 'Alfa_amilasa': -0.05,
    'PCR': 0.05, 'Hemoglobina_libre': 0.33, 'Magnesio': 0.43,
    'Gamma_GT': 0.44, 'MCH': 0.46, 'Colesterol': 0.48
}

# Medianas del Grupo Control (0.0) para imputación segura
MEDIANAS_CONTROL = {
    'Trigliceridos': 113.0, 'Glucosa': 107.0, 'Colinesterasa': 7.4,
    'Cloruro': 103.0, 'Albumina': 41.7, 'Alfa_amilasa': 57.0,
    'PCR': 0.5, 'Hemoglobina_libre': 4.1, 'Magnesio': 0.8,
    'Gamma_GT': 32.0, 'MCH': 29.7, 'Colesterol': 176.0
}

CORE_VARS = ['Glucosa', 'Trigliceridos', 'Colinesterasa', 'Gamma_GT', 'MCH']

def procesar_analitica_paciente(datos_paciente):
    paciente_df = pd.Series(datos_paciente)

    nulos_totales = paciente_df.isna().sum()
    nulos_core = paciente_df[CORE_VARS].isna().sum()

    if nulos_totales > 3 or nulos_core > 2:
        return {
            "estado": "ERROR",
            "mensaje": f"Datos insuficientes. Faltan {nulos_totales} valores en total y {nulos_core} principales. Máximo permitido: 3 globales y 2 principales.",
            "datos_procesados": None,
            "imputados": []
        }

    variables_imputadas = []
    paciente_procesado = paciente_df.copy()

    for var in PESOS.keys():
        if pd.isna(paciente_procesado[var]):
            paciente_procesado[var] = MEDIANAS_CONTROL[var]
            variables_imputadas.append(var)

    mensaje_exito = "Datos procesados correctamente."
    if variables_imputadas:
        mensaje_exito += f" Se han imputado valores de control sanos en: {', '.join(variables_imputadas)}."

    return {
        "estado": "OK",
        "mensaje": mensaje_exito,
        "datos_procesados": paciente_procesado,
        "imputados": variables_imputadas
    }
