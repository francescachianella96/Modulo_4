import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. CONFIGURAZIONE INPUT ---
# Immagine 28x28 a scala di grigio (1 canale)
input_shape = (28, 28, 1) 

# --- 2. APPROCCIO ANN (LAYER DENSO) ---
# Dobbiamo appiattire l'immagine in un vettore di 784 elementi
model_dense = models.Sequential([
    layers.Flatten(input_shape=input_shape),
    layers.Dense(32, activation='relu', name="Layer_Denso")
], name="Architettura_ANN")

# --- 3. APPROCCIO CNN (LAYER CONVOLUZIONALE) ---
# Manteniamo la struttura 2D. Usiamo 32 filtri da 3x3
model_conv = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', name="Layer_Convoluzionale")
], name="Architettura_CNN")

# --- 4. CONFRONTO ANALITICO ---
print("CONFRONTO PARAMETRI:")
print(f"{'Modello':<25} | {'Parametri Totali':<20}")
print("-" * 50)
print(f"{model_dense.name:<25} | {model_dense.count_params():<20}")
print(f"{model_conv.name:<25} | {model_conv.count_params():<20}")

# Ispezione dettagliata
print("\nDETTAGLIO ANN:")
model_dense.summary()

print("\nDETTAGLIO CNN:")
model_conv.summary()