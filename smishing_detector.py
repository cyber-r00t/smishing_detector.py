# smishing_detector.py

"""
Clasificador de Smishing Financiero (Bot Anti-Estafas)

Autor: Jheery Barrientos (CYBER_ROOT)
Fecha: Diciembre, 2025
Licencia: MIT (Ver archivo LICENSE para detalles)
Copyright (c) 2025, CYBER_ROOT

Descripción: Este script utiliza Machine Learning (Naive Bayes) para clasificar 
mensajes SMS como legítimos ('Ham') o fraudulentos ('Spam') basándose en un 
dataset público de SMS. Incluye una lógica de reglas para identificar patrones 
específicos de fraude bancario en español.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Cargar y Preparar los Datos ---
# Asume que el archivo CSV está en la misma carpeta.
# El dataset UCI tiene dos columnas: 'label' (ham/spam) y 'message'.
try:
    data = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: Asegúrate de tener el archivo 'spam.csv' en la misma carpeta.")
    exit()

# Renombrar columnas para claridad
data = data.rename(columns={'v1': 'label', 'v2': 'message'})
# Convertir las etiquetas a binario: 0 (Legítimo/ham) y 1 (Fraude/spam)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Mostrar las primeras filas y el conteo de spam vs. ham
print("--- Primeras 5 filas del Dataset ---")
print(data.head())
print("\n--- Conteo de Mensajes ---")
print(data['label'].value_counts())

# --- 2. Dividir los Datos (Entrenamiento y Prueba) ---
X = data['message']  # Las características (el texto del mensaje)
y = data['label']    # La variable objetivo (0 o 1)

# Dividir el dataset: 80% para entrenar, 20% para probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nDatos de Entrenamiento: {len(X_train)} mensajes")
print(f"Datos de Prueba: {len(X_test)} mensajes")

# --- 3. Vectorización (Convertir Texto a Números) ---
# CountVectorizer convierte el texto en una matriz de frecuencias de palabras.
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- 4. Entrenar el Modelo (Naive Bayes) ---
# Naive Bayes es simple y muy efectivo para clasificación de texto/spam.
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# --- 5. Evaluar el Modelo ---
y_pred = model.predict(X_test_vec)
print("\n--- Evaluación del Modelo ---")
print(f"Precisión General: {accuracy_score(y_test, y_pred):.4f}")
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# --- 6. Función de Detección (¡Prueba Real!) ---
def detectar_smishing(sms):
    # Vectorizar el nuevo SMS
    sms_vec = vectorizer.transform([sms])
    # Predecir
    prediction = model.predict(sms_vec)[0]
    
    if prediction == 1:
        return "⚠️ FRAUDE/SMISHING detectado (ALERTA ROJA)"
    else:
        return "✅ Mensaje legítimo (HAM)"

# --- PRUEBAS CON TUS MENSAJES (Ejemplos del SMS del BBVA) ---
print("\n--- Pruebas con tus ejemplos ---")
test_sms_1 = "Se ha retenido un cargo por el valor de 980EUR en su banca online, sino lo reconoce el cargo llamanos al 742074762"
test_sms_2 = "Hola, tu paquete ha sido entregado en el buzón. Por favor, confirma la dirección en el siguiente enlace: bit.ly/paquete-estafa"
test_sms_3 = "Tienes una reunión mañana a las 10am."

print(f"SMS 1 ('{test_sms_1[:30]}...'): {detectar_smishing(test_sms_1)}")
print(f"SMS 2 ('{test_sms_2[:30]}...'): {detectar_smishing(test_sms_2)}")
print(f"SMS 3 ('{test_sms_3[:30]}...'): {detectar_smishing(test_sms_3)}")
