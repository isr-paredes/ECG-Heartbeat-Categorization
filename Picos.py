import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV (modifica 'archivo.csv' con el nombre de tu archivo)
df = pd.read_csv('mitbih_train.csv')

# Selecciona una fila específica (por ejemplo, la primera fila) y excluye la última columna
fila = df.iloc[1, :-1].values  # Esto selecciona la primera fila y excluye la última columna

# Calcular la primera derivada utilizando diferencias finitas
primera_derivada = np.diff(fila)

# Calcular la segunda derivada utilizando diferencias finitas sobre la primera derivada
segunda_derivada = np.diff(primera_derivada)
#segunda_derivada = np.diff(fila)

# Solicitar al usuario que ingrese el valor umbral
umbral = 0.05

# Contar la cantidad de veces que la segunda derivada supera el umbral
contador = np.sum(np.abs(segunda_derivada) > umbral)

print(f"La segunda derivada supera el valor {umbral} un total de {contador} veces.")

# Opcional: Graficar la segunda derivada
plt.figure(figsize=(10, 6))
plt.plot(segunda_derivada)
plt.axhline(y=umbral, color='r', linestyle='--', label='Umbral')
plt.axhline(y=-umbral, color='r', linestyle='--')
plt.title('Segunda Derivada de la Señal')
plt.xlabel('Índice')
plt.ylabel('Segunda Derivada')
plt.legend()
plt.show()
