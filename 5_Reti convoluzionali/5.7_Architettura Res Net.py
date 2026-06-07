import tensorflow as tf
from tensorflow.keras import layers, models

def residual_block(input_tensor, filters, kernel_size=3):
    """
    LOGICA DELLA DERIVATA (Backpropagation):
    1. Forward Pass: H(x) = F(x) + x 
       (dove F(x) è il percorso convoluzionale e x è lo shortcut)
    
    2. Backward Pass (Derivata rispetto a x): 
       dH/dx = d/dx[F(x)] + d/dx[x] 
       dH/dx = F'(x) + 1
    
    3. Conclusione: Quel "+ 1" è l'autostrada. Anche se i pesi nelle convoluzioni
       portano F'(x) vicino a zero (Vanishing Gradient), il gradiente totale 
       dH/dx rimarrà almeno 1, permettendo al segnale di tornare ai layer iniziali.
    """
    
    shortcut = input_tensor

    # --- PRIMO LAYER ---
    # Calcolo Parametri: ((3*3 * 64 in) + 1 bias) * 64 filtri = 36.928
    x = layers.Conv2D(filters, kernel_size, padding='same')(input_tensor)
    
    # Calcolo Parametri BN: 4 * 64 filtri (gamma, beta, mean, var) = 256
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # --- SECONDO LAYER ---
    # Calcolo Parametri: ((3*3 * 64 in) + 1 bias) * 64 filtri = 36.928
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    
    # Calcolo Parametri BN: 4 * 64 filtri = 256
    x = layers.BatchNormalization()(x)

    # --- LA MAGIA DI RESNET ---
    # Somma: H(x) = F(x) + x (Operazione element-wise, 0 parametri)
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

# --- CONFIGURAZIONE E ISPEZIONE ---
# Input con 64 canali
inputs = layers.Input(shape=(32, 32, 64))
outputs = residual_block(inputs, filters=64)
model = models.Model(inputs=inputs, outputs=outputs, name="ResNet_Single_Block")

# PARAMETRI TOTALI PREVISTI:
# Conv1(36928) + BN1(256) + Conv2(36928) + BN2(256) = 74.368
model.summary()