import numpy as np

# Datos extraídos de la Tabla 1 (Medianas del grupo control 0.0) para imputación
MEDIANAS_CONTROL = {
    'Trigliceridos': 113.0,
    'Glucosa': 107.0,
    'Colinesterasa': 7.4,
    'Cloruro': 103.0,
    'Albumina': 4.1,
    'Alfa_amilasa': 57.0,
    'PCR': 1.6,
    'Hb_libre': 4.1,
    'Magnesio': 0.8,
    'Gamma_GT': 32.0,
    'MCH': 29.7,
    'Colesterol': 176.0
}

# Pesos de la regresión logística (Feature Selection con XGBoost)
COEFICIENTES = {
    'Trigliceridos': -0.65,
    'Glucosa': -0.63,
    'Colinesterasa': -0.48,
    'Cloruro': -0.38,
    'Albumina': -0.22,
    'Alfa_amilasa': -0.05,
    'PCR': 0.05,
    'Hb_libre': 0.33,
    'Magnesio': 0.43,
    'Gamma_GT': 0.44,
    'MCH': 0.46,
    'Colesterol': 0.48
}

VARIABLES_CORE = ['Glucosa', 'Trigliceridos', 'Colinesterasa', 'Gamma_GT', 'MCH']

def procesar_analitica_paciente(datos_paciente):
    """
    Filtro de calidad e imputación de valores faltantes basado en las medianas de control.
    """
    faltantes_totales = 0
    faltantes_core = 0
    datos_procesados = {}
    valores_imputados = []

    for clave, valor in datos_paciente.items():
        if valor is None or valor == 0.0:
            faltantes_totales += 1
            if clave in VARIABLES_CORE:
                faltantes_core += 1
            datos_procesados[clave] = MEDIANAS_CONTROL[clave]
            valores_imputados.append(clave)
        else:
            datos_procesados[clave] = valor

    # Regla estricta de bloqueo
    if faltantes_totales > 3 or faltantes_core > 2:
        return {
            'estado': 'ERROR',
            'mensaje': f'Datos insuficientes. Faltan {faltantes_totales} valores en total y {faltantes_core} principales. Máximo permitido: 3 globales y 2 principales.',
            'datos_procesados': None
        }
    
    mensaje_exito = "Datos procesados correctamente."
    if valores_imputados:
        mensaje_exito += f" Se han imputado valores de control sanos en: {', '.join(valores_imputados)}."

    return {
        'estado': 'OK',
        'mensaje': mensaje_exito,
        'datos_procesados': datos_procesados
    }

def calcular_score_bruto(datos_procesados):
    """
    Realiza la ecuación de combinación lineal: suma el producto de cada valor por su coeficiente.
    Nota: Falta añadir el intercepto local de Jaén en un futuro.
    """
    score_bruto = 0.0
    for clave, valor in datos_procesados.items():
        score_bruto += (valor * COEFICIENTES[clave])
    
    return round(score_bruto, 2)

def evaluar_perfil_riesgo(datos_procesados):
    """
    Genera alertas descriptivas basadas en el comportamiento fisiopatológico.
    """
    alertas = []
    if datos_procesados['Glucosa'] < 90:
        alertas.append(f"Glucosa Baja ({datos_procesados['Glucosa']}) - Paradoja metabólica")
    if datos_procesados['Trigliceridos'] < 100:
        alertas.append(f"Triglicéridos Bajos ({datos_procesados['Trigliceridos']}) - Paradoja metabólica")
    if datos_procesados['Colinesterasa'] < 6.0:
        alertas.append(f"Colinesterasa Baja ({datos_procesados['Colinesterasa']}) - Riesgo hepático")
    if datos_procesados['Gamma_GT'] > 40:
        alertas.append(f"Gamma-GT Elevada ({datos_procesados['Gamma_GT']}) - Congestión retrógrada")
    if datos_procesados['MCH'] > 32:
        alertas.append(f"MCH Elevada ({datos_procesados['MCH']}) - Patrón hematológico")
        
    return alertas
