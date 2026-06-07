import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. CONFIGURAZIONE INPUT ---
# Immagine 32x32 a colori (3 canali RGB)
input_shape = (32, 32, 3) 

# --- 2. APPROCCIO CNN (Esercizio RGB) ---
# CALCOLO PARAMETRI:
# Ogni filtro deve "attraversare" tutti i canali dell'input (3).
# 1. Pesi per filtro: (5 * 5 * 3) = 75 pesi
# 2. Bias per filtro: 1
# 3. Parametri per singolo filtro: 75 + 1 = 76
# 4. Totale per 64 filtri: 76 * 64 = 4864
model_conv = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(64, kernel_size=(5, 5), activation='relu', name="Layer_RGB_64_Filtri")
], name="Architettura_CNN_Colori")

# --- 3. VERIFICA ---
print(f"PARAMETRI TOTALI CALCOLATI: 4864")
print(f"PARAMETRI TOTALI DA KERAS: {model_conv.count_params()}")
print("-" * 50)
print("\nDETTAGLIO ARCHITETTURA:")
model_conv.summary()