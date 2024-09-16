import numpy as np
import pandas as pd
import joblib

# Cargar el modelo entrenado desde el archivo
modelo = joblib.load('Clasificador.pkl')

# Cargar la nueva muestra desde un archivo CSV (sin la columna de categoría)
nueva_muestra = pd.read_csv('Muestras/Muestra_cat3.csv', header=None)

# Obtener las probabilidades de cada clase
probabilidades = modelo.predict_proba(nueva_muestra)

# Mostrar la probabilidad para cada clase
print("Probabilidades de cada clase:", probabilidades[0])

# Mostrar la clase con la mayor probabilidad
print("La clase predicha para la nueva muestra es:", modelo.classes_[probabilidades[0].argmax()])