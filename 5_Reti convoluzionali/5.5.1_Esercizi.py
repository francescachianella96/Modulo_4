import tensorflow as tf
from tensorflow.keras import layers, models

def build_alexnet(input_shape=(227, 227, 3), num_classes=1000):
    model = models.Sequential([
        # --- 1. INPUT LAYER ---
        layers.Input(shape=input_shape),

        # --- LAYER 1 ---
        # Conv1 Output: floor((120 - 11) / 4) + 1 = 28 -> (28, 28, 96)
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', name='Conv1'),
        
        # Pool1 Output: floor((28 - 3) / 2) + 1 = 13 -> (13, 13, 96)
        # RISPOSTA QUESITO 1: La dimensione dopo Pool1 è (13, 13, 96)
        layers.MaxPooling2D(pool_size=(3, 3), strides=2, name='Pool1'),

        # --- LAYER 2 ---
        # Conv2 (Padding 'same'): (13, 13, 256)
        layers.Conv2D(256, (5, 5), padding='same', activation='relu', name='Conv2'),
        # Pool2: floor((13 - 3) / 2) + 1 = 6 -> (6, 6, 256)
        layers.MaxPooling2D(pool_size=(3, 3), strides=2, name='Pool2'),

        # --- LAYER 3, 4, 5 ---
        layers.Conv2D(384, (3, 3), padding='same', activation='relu', name='Conv3'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu', name='Conv4'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='Conv5'),
        # Pool3: floor((6 - 3) / 2) + 1 = 2 -> (2, 2, 256)
        layers.MaxPooling2D(pool_size=(3, 3), strides=2, name='Pool3'),

        # --- CLASSIFICATORE ---
        # Flatten: 2 * 2 * 256 = 1024 neuroni (molto meno rispetto ai 9216 originali)
        layers.Flatten(),
        
        # FC1: (1024 + 1) * 4096 = 4.198.400 parametri
        layers.Dense(4096, activation='relu', name='FC1'),
        layers.Dropout(0.5),
        
        # FC2: (4096 + 1) * 4096 = 16.781.312 parametri
        layers.Dense(4096, activation='relu', name='FC2'),
        layers.Dropout(0.5),
        
        # Output: (4096 + 1) * 10 = 40.970 parametri
        layers.Dense(num_classes, activation='softmax', name='Output')
        
    ], name="AlexNet_Ridotta")
    
    return model

# --- MODIFICHE RICHIESTE DALLA TRACCIA ---
# 1. input_shape portata a (120, 120, 3)
# 2. num_classes portato a 10
alex_model_ridotta = build_alexnet(input_shape=(120, 120, 3), num_classes=10)

# Visualizzazione dei risultati
alex_model_ridotta.summary()

# RISPOSTA QUESITO 2: 
# Parametri Totali: 24,767,882 (rispetto ai ~62 milioni dell'originale)