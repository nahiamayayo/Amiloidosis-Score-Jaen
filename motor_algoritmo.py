import numpy as np

# 10 biomarcadores oficiales del Protocolo CardioGen Jaén
BIOMARCADORES_OFICIALES = [
    'Glucosa', 'Trigliceridos', 'Colesterol', 'Gamma_GT', 'Albumina',
    'MCH', 'Magnesio', 'Hb_libre', 'PCR', 'proBNP'
]

# Valores de referencia de la cohorte control de Jaén para imputación (Medianas locales)
MEDIANAS_JAEN = {
    'Glucosa': 94.0,
    'Trigliceridos': 110.0,
    'Colesterol': 180.0,
    'Gamma_GT': 28.0,
    'Albumina': 4.2,
    'MCH': 30.0,
    'Magnesio': 0.85,
    'Hb_libre': 2.5,
    'PCR': 1.2,
    'proBNP': 150.0
}

# Rangos de normalidad del Laboratorio Central del HUJ (Hospital Universitario de Jaén)
RANGOS_REFERENCIA_HUJ = {
    'Glucosa': (70.0, 99.0),        # mg/dL
    'Trigliceridos': (0.0, 150.0),   # mg/dL
    'Colesterol': (120.0, 200.0),   # mg/dL
    'Gamma_GT': (0.0, 61.0),         # U/L
    'Albumina': (3.5, 5.0),         # g/dL
    'MCH': (27.0, 33.0),            # pg
    'Magnesio': (1.6, 2.6),         # mg/dL
    'Hb_libre': (0.0, 5.0),         # mg/dL
    'PCR': (0.0, 5.0),              # mg/L
    'proBNP': (0.0, 300.0)          # pg/mL
}

def procesar_y_validar_analitica(datos_paciente):
    """
    Pipeline de control de calidad e imputación estadística adaptativa.
    Garantiza que el número de datos faltantes (NaN, None o 0.0) no supere
    la barrera de seguridad analítica institucional (máximo 2 parámetros ausentes).
    """
    faltantes_totales = 0
    datos_procesados = {}
    biomarcadores_reemplazados = []

    for biomarcador in BIOMARCADORES_OFICIALES:
        valor = datos_paciente.get(biomarcador, None)

        # Identificación estricta de datos faltantes o nulos
        if valor is None or valor == 0.0 or (isinstance(valor, float) and np.isnan(valor)):
            faltantes_totales += 1
            datos_procesados[biomarcador] = MEDIANAS_JAEN[biomarcador]
            biomarcadores_reemplazados.append(biomarcador)
        else:
            datos_procesados[biomarcador] = float(valor)

    # Barrera de seguridad analítica (Bloqueo fulminante si faltan más de 2 parámetros)
    if faltantes_totales > 2:
        return {
            'estado': 'BLOQUEO',
            'mensaje': f'⚠️ BLOQUEO DE SEGURIDAD ANÁLITICA: Se han detectado {faltantes_totales} parámetros ausentes. El protocolo exige un mínimo de 8 parámetros reales.',
            'datos_listos': None
        }

    mensaje = "Datos procesados correctamente."
    if biomarcadores_reemplazados:
        mensaje += f" Imputación estadística adaptativa aplicada en: {', '.join(biomarcadores_reemplazados)}."

    return {
        'estado': 'OK',
        'mensaje': mensaje,
        'datos_listos': datos_procesados
    }

def calcular_probabilidad_score(datos_listos, datos_modelo):
    """
    Calcula la probabilidad predictiva final utilizando el modelo calibrado de Jaén.
    Realiza la estandarización estricta (Z-score) previa a la combinación lineal.
    """
    if datos_modelo is None:
        raise ValueError("Se requiere el diccionario con los objetos del modelo entrenado (.pkl) para extraer la configuración de Jaén.")

    clf = datos_modelo['clf']
    scaler = datos_modelo['scaler']
    columnas = datos_modelo['columnas']

    # Ordenar el vector del paciente exactamente en el mismo orden de variables del entrenamiento
    vector_paciente = [datos_listos[col] for col in columnas]
    X_array = np.array(vector_paciente).reshape(1, -1)

    # 1. Estandarización rigurosa (Z-score) utilizando la media y desviación típica de la cohorte local
    X_scaled = scaler.transform(X_array)

    # 2. Evaluación mediante el modelo lineal y obtención de la probabilidad (curva sigmoide)
    probabilidad = clf.predict_proba(X_scaled)[0][1]

    return probabilidad

def evaluar_alertas_clinicas_huj(datos_listos):
    """
    Genera alertas diagnósticas auxiliares basadas de forma estricta en los rangos
    de normalidad del Laboratorio Central del Hospital Universitario de Jaén (HUJ).
    """
    alertas = []
    
    for biomarcador, (min_ref, max_ref) in RANGOS_REFERENCIA_HUJ.items():
        valor = datos_listos.get(biomarcador, None)
        if valor is not None:
            if valor < min_ref or valor > max_ref:
                alertas.append(f"{biomarcador} fuera de rango referencial ({min_ref} - {max_ref})")
                
    return alertas
