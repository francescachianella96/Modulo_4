import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(input_tensor, filters, kernel_size=3):
    shortcut = input_tensor
    
    # --- MODIFICA: Controllo dimensionalità (Projection Shortcut) ---
    # Se i canali in ingresso sono diversi dai filtri in uscita,
    # dobbiamo adattare la shortcut con una conv 1x1.
    input_channels = input_tensor.shape[-1]
    
    if input_channels != filters:
        # Calcolo Parametri Conv 1x1: ((1*1 * 64 in) + 1 bias) * 128 filtri = 8.320
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(input_tensor)
        # Calcolo Parametri BN: 4 * 128 = 512
        shortcut = layers.BatchNormalization()(shortcut)

    # --- PERCORSO CONVOLUZIONALE (Main Path) ---
    # Primo layer: ((3*3 * 64 in) + 1 bias) * 128 filtri = 73.856
    x = layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Secondo layer: ((3*3 * 128 in) + 1 bias) * 128 filtri = 147.584
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # --- SOMMA ---
    # Ora x e shortcut hanno entrambi 128 canali.
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

# --- CONFIGURAZIONE E ISPEZIONE ---
# Input con 64 canali, ma chiediamo 128 filtri in uscita
inputs = layers.Input(shape=(32, 32, 64))
outputs = residual_block(inputs, filters=128)
model = models.Model(inputs=inputs, outputs=outputs, name="ResNet_Projection_Block")

model.summary()