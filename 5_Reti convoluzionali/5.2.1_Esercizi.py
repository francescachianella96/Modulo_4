import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# --- 1. SIMULAZIONE INPUT ---
# Modificato: dimensione 64x64 come da esercizio
input_image = np.random.rand(1, 64, 64, 3).astype(np.float32)

# --- 2. DEFINIZIONE LAYER CONVOLUZIONALE ---
model = models.Sequential([
    # Modificato: shape (64, 64, 3)
    layers.Input(shape=(64, 64, 3)),
    
    # AGGIUNTO: Cornice di 2 zeri per lato (Padding: 2)
    layers.ZeroPadding2D(padding=2),
    
    # LAYER ESERCIZIO: 
    # Formula: floor((Input_finito(68) - kernel(5)) / stride(2)) + 1
    # floor(63 / 2) + 1 = 31 + 1 = 32
    # Output: (32, 32, 32)
    layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu')
])

# --- 3. ANALISI DIMENSIONI ---
model.summary()

# Risultato atteso: 32x32x32