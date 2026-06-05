import pickle
import numpy as np
import pandas as pd

print("\n" + "="*50)
print("🩺 AUDITORÍA MATEMÁTICA - CARDIOGEN JAÉN")
print("="*50)

# 1. Cargar el modelo institucional (el cerebro de la IA)
try:
    with open('modelo_jaen.pkl', 'rb') as f:
        datos = pickle.load(f)
    clf = datos['clf']
    imputer = datos['imputer']
    scaler = datos['scaler']
    columnas = datos['columnas']
except FileNotFoundError:
    print("❌ Error: No se encuentra 'modelo_jaen.pkl'. Calibra el modelo en la web primero.")
    exit()

# 2. Introduce aquí los datos en crudo del paciente que quieras auditar
paciente_prueba = {
    'Glucosa': 105.0,
    'Trigliceridos': 150.0,
    'Colesterol': 210.0,
    'Gamma_GT': 45.0,
    'Albumina': 3.8,
    'MCH': 31.0,
    'Magnesio': 1.9,
    'Hb_libre': 12.5,
    'PCR': 6.2,
    'proBNP': 1200.0  # Dato clave para ver cómo sube el riesgo
}

# Convertir a formato vector (ordenado igual que el entrenamiento)
valores_crudos = [paciente_prueba[col] for col in columnas]
X_crudo = np.array(valores_crudos).reshape(1, -1)

# 3. PROCESAMIENTO MATEMÁTICO PASO A PASO
print("\n--- 1. ESTANDARIZACIÓN (Z-SCORE) ---")
X_imp = imputer.transform(X_crudo)
X_sca = scaler.transform(X_imp)[0] # Extraemos la matriz transformada

for i, col in enumerate(columnas):
    print(f"  {col}: Valor Crudo = {valores_crudos[i]} --> Valor Estandarizado (Z) = {X_sca[i]:.4f}")

print("\n--- 2. ECUACIÓN LOGIT (Combinación Lineal) ---")
intercepto = clf.intercept_[0]
coeficientes = clf.coef_[0]

print(f"  Intercepto (β0) = {intercepto:.4f}")
logit_total = intercepto

for i, col in enumerate(columnas):
    impacto_variable = X_sca[i] * coeficientes[i]
    logit_total += impacto_variable
    print(f"  + {col}: (Z: {X_sca[i]:.4f} * Peso: {coeficientes[i]:.4f}) = {impacto_variable:.4f}")

print("-" * 30)
print(f"  RESULTADO LOGIT TOTAL = {logit_total:.4f}")

print("\n--- 3. FUNCIÓN SIGMOIDE (Transformación a Probabilidad) ---")
# Fórmula matemática de la sigmoide
probabilidad_calculada = 1 / (1 + np.exp(-logit_total))
print(f"  Fórmula: 1 / (1 + e^-({logit_total:.4f}))")
print(f"  PROBABILIDAD MATEMÁTICA MANUAL: {probabilidad_calculada:.2%}")

# 4. Verificación cruzada con la función de la librería
probabilidad_sklearn = clf.predict_proba(X_sca.reshape(1, -1))[0][1]
print(f"  PROBABILIDAD DEL ALGORITMO WEB: {probabilidad_sklearn:.2%}")

if np.isclose(probabilidad_calculada, probabilidad_sklearn):
    print("\n✅ VERIFICACIÓN SUPERADA: Los cálculos manuales coinciden exactamente con el algoritmo.")
else:
    print("\n❌ ALERTA: Discrepancia matemática detectada.")
print("="*50 + "\n")
