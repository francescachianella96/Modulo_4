import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# --- 1. SIMULAZIONE INPUT ---
# Creiamo un'immagine fittizia 32x32 con 3 canali (RGB)
# La forma richiesta è (Batch, Altezza, Larghezza, Canali)
input_image = np.random.rand(1, 32, 32, 3).astype(np.float32)

# --- 2. DEFINIZIONE LAYER CONVOLUZIONALE ---
model = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    
    # LAYER 1: 
    # Formula 'same': 32 / stride(1) = 32
    # Output: (32, 32, 16)
    layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'),
    
    # LAYER 2:
    # Formula 'valid': floor((32 - kernel(5)) / stride(2)) + 1
    # floor(27 / 2) + 1 = 13 + 1 = 14
    # Output: (14, 14, 32)
    layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation='relu')
])

# --- 3. ANALISI DIMENSIONI (Punto chiave 3) ---
# Visualizziamo come cambiano le feature maps
model.summary()

# Output previsto per il primo layer (same padding): 32x32x16
# Output previsto per il secondo layer (valid padding, stride 2): 14x14x32