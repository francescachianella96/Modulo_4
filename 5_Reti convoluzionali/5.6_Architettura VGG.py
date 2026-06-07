import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. CONFIGURAZIONE INPUT ---
input_shape = (32, 32, 3)

# --- 2. BLOCCO IN STILE VGG (Due 3x3) ---
# Ogni layer aggiunge una ReLU, aumentando la capacità di apprendimento
vgg_block = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2))
], name="VGG_Style_Block")

# --- 3. BLOCCO CON KERNEL GRANDE (Singolo 5x5) ---
# Un solo layer, una sola attivazione ReLU
large_kernel_block = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu'),
    layers.Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2))
], name="Large_Kernel_Block")

# --- 4. ANALISI COMPARATIVA ---
print(f"Parametri Blocco VGG (Due 3x3): {vgg_block.count_params()}")
print(f"Parametri Blocco Kernel Grande (5x5): {large_kernel_block.count_params()}")

# Ispezione della gerarchia
vgg_block.summary()