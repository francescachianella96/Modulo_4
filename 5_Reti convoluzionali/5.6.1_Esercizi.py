import tensorflow as tf
from tensorflow.keras import layers, models

# --- 1. CONFIGURAZIONE INPUT ---
input_shape = (32, 32, 3)

# --- 2. COSTRUZIONE VGG-4 ---
vgg4_model = models.Sequential([
    layers.Input(shape=input_shape),
    
    # BLOCCO 1: 2 layer Conv 3x3 (64 filtri) + MaxPooling
    # Calcolo parametri Conv1: ((3*3*3)+1)*64 = 1.792
    # Calcolo parametri Conv2: ((3*3*64)+1)*64 = 36.928
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name="Conv1_1"),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', name="Conv1_2"),
    layers.MaxPooling2D(pool_size=(2, 2), name="Pool1"), # Output: (16, 16, 64)

    # BLOCCO 2: 2 layer Conv 3x3 (128 filtri) + MaxPooling
    # Calcolo parametri Conv3: ((3*3*64)+1)*128 = 73.856
    # Calcolo parametri Conv4: ((3*3*128)+1)*128 = 147.584
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name="Conv2_1"),
    layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu', name="Conv2_2"),
    layers.MaxPooling2D(pool_size=(2, 2), name="Pool2"), # Output: (8, 8, 128)

    # TESTA DEL MODELLO (Classificazione)
    layers.Flatten(name="Flatten"), # 8*8*128 = 8.192 neuroni
    # Calcolo parametri Dense: (8.192 + 1) * 10 = 81.930
    layers.Dense(10, activation='softmax', name="Output_Dense")
], name="VGG-4_Reduced")

# --- 3. ANALISI DEI PARAMETRI ---
conv_params = 1792 + 36928 + 73856 + 147584
dense_params = 81930

print(f"Parametri Parte Convoluzionale: {conv_params:,}")
print(f"Parametri Parte Densa: {dense_params:,}")
print("-" * 30)

vgg4_model.summary()