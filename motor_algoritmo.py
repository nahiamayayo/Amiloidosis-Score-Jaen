import numpy as np

# Datos de referencia para imputación (Medianas del grupo control original)
MEDIANAS_CONTROL = {
    'Trigliceridos': 113.0, 'Glucosa': 107.0, 'Colinesterasa': 7.4,
    'Cloruro': 103.0, 'Albumina': 4.1, 'Alfa_amilasa': 57.0,
    'PCR': 1.6, 'Hb_libre': 4.1, 'Magnesio': 0.8,
    'Gamma_GT': 32.0, 'MCH': 29.7, 'Colesterol': 176.0
}

# Pesos iniciales del modelo (estudio base)
COEFICIENTES_BASE = {
    'Trigliceridos': -0.65, 'Glucosa': -0.63, 'Colinesterasa': -0.48,
    'Cloruro': -0.38, 'Albumina': -0.22, 'Alfa_amilasa': -0.05,
    'PCR': 0.05, 'Hb_libre': 0.33, 'Magnesio': 0.43,
    'Gamma_GT': 0.44, 'MCH': 0.46, 'Colesterol': 0.48
}

VARIABLES_CORE = ['Glucosa', 'Trigliceridos', 'Colinesterasa', 'Gamma_GT', 'MCH']

def procesar_analitica_paciente(datos_paciente):
    """
    Filtro de calidad e imputación.
    """
    faltantes_totales = 0
    faltantes_core = 0
    datos_procesados = {}
    valores_imputados = []

    for clave, valor in datos_paciente.items():
        # Limpieza: si valor es None, 0 o NaN (np.nan), imputamos
        if valor is None or valor == 0.0 or (isinstance(valor, float) and np.isnan(valor)):
            faltantes_totales += 1
            if clave in VARIABLES_CORE:
                faltantes_core += 1
            datos_procesados[clave] = MEDIANAS_CONTROL.get(clave, 0.0)
            valores_imputados.append(clave)
        else:
            datos_procesados[clave] = valor

    if faltantes_totales > 3 or faltantes_core > 2:
        return {
            'estado': 'ERROR',
            'mensaje': f'Datos insuficientes. Faltan {faltantes_totales} valores. Máximo permitido: 3.',
            'datos_procesados': None
        }
    
    mensaje_exito = "Datos procesados correctamente."
    if valores_imputados:
        mensaje_exito += f" Valores imputados en: {', '.join(valores_imputados)}."

    return {'estado': 'OK', 'mensaje': mensaje_exito, 'datos_procesados': datos_procesados}

def calcular_score_bruto(datos_procesados, coeficientes=None):
    """
    Realiza la ecuación de combinación lineal. 
    Permite pasar coeficientes personalizados desde el modelo entrenado en Jaén.
    """
    coefs = coeficientes if coeficientes is not None else COEFICIENTES_BASE
    score_bruto = 0.0
    for clave, valor in datos_procesados.items():
        score_bruto += (valor * coefs.get(clave, 0.0))
    
    return round(score_bruto, 2)

def evaluar_perfil_riesgo(datos):
    """
    Alertas clínicas basadas en los rangos del Hospital de Jaén.
    """
    alertas = []
    if datos.get('Glucosa', 0) > 99 or (0 < datos.get('Glucosa', 0) < 70):
        alertas.append("Glucosa fuera de rango (70-99 mg/dL)")
    if datos.get('Trigliceridos', 0) > 150:
        alertas.append("Triglicéridos elevados (>150 mg/dL)")
    if datos.get('Gamma_GT', 0) > 61:
        alertas.append("Gamma-GT elevada (>61 U/L)")
    if datos.get('PCR', 0) > 5:
        alertas.append("PCR elevada (>5 mg/L)")
    # ... resto de tus reglas clínicas aquí
        
    return alertas
