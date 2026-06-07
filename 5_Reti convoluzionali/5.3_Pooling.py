import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# --- 1. PREPARAZIONE DATI (Punto chiave 3) ---
# Creiamo una matrice 4x4 di prova per simulare una Feature Map
# Aggiungiamo le dimensioni per Batch e Canali: (1, 4, 4, 1)
data = np.array([
    [10, 20, 30, 40],
    [15, 25, 35, 45],
    [50, 60, 70, 80],
    [55, 65, 75, 95]
], dtype=np.float32).reshape(1, 4, 4, 1)

# --- 2. MAX POOLING (Punto chiave 2) ---
# Estrae il valore massimo in ogni finestra 2x2
max_pool_layer = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
max_output = max_pool_layer(data)

# --- 3. AVERAGE POOLING (Punto chiave 2) ---
# Calcola la media in ogni finestra 2x2
avg_pool_layer = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
avg_output = avg_pool_layer(data)

print("Matrice Originale 4x4:\n", data.reshape(4, 4))
print("\nRisultato Max Pooling 2x2 (Sintesi dei picchi):\n", max_output.numpy().reshape(2, 2))
print("\nRisultato Average Pooling 2x2 (Sintesi media):\n", avg_output.numpy().reshape(2, 2))