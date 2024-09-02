import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV (modifica 'archivo.csv' con el nombre de tu archivo)
df = pd.read_csv('mitbih_train.csv')

# Selecciona una fila específica (por ejemplo, la primera fila) y excluye la última columna
fila = df.iloc[1, :-1].values  # Esto selecciona la primera fila y excluye la última columna

num_frecuencias = 200

# Realiza la Transformada de Fourier
transformada = np.fft.fft(fila,num_frecuencias)

# Calcula las frecuencias asociadas
frecuencias = np.fft.fftfreq(num_frecuencias, d=(1/125))

# Grafica el espectro de frecuencias
plt.figure(figsize=(10, 6))
print(frecuencias)
print(transformada)
plt.plot(frecuencias, np.abs(transformada))
plt.title('Transformada de Fourier')
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.show()
