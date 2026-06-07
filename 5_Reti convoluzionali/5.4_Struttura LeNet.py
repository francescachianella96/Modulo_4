import tensorflow as tf
from tensorflow.keras import layers, models

def build_lenet_modern(input_shape=(32, 32, 1)):
    """
    Costruisce l'architettura LeNet-5 con sintassi moderna.
    
    CALCOLO PARAMETRI (Esempio):
    Formula Conv2D: ((Kernel_W * Kernel_H * Input_Channels) + 1) * Filters
    Formula Dense:  (Input_Units + 1) * Output_Units
    """
    
    model = models.Sequential([
        # --- INPUT LAYER ---
        # Definiamo esplicitamente l'ingresso (Batch, 32, 32, 1)
        layers.Input(shape=input_shape),

        # --- PRIMO BLOCCO CONVOLUZIONALE ---
        # Calcolo: ((5 * 5 * 1) + 1) * 6 = 26 * 6 = 156 parametri
        # Output spaziale (Valid): (32 - 5)/1 + 1 = 28 -> (28, 28, 6)
        layers.Conv2D(6, kernel_size=(5, 5), activation='relu', name="Conv1"),
        layers.AveragePooling2D(pool_size=(2, 2), name="Pool1"), # Dim: (14, 14, 6)

        # --- SECONDO BLOCCO CONVOLUZIONALE ---
        # Calcolo: ((5 * 5 * 6) + 1) * 16 = (150 + 1) * 16 = 2416 parametri
        # Output spaziale (Valid): (14 - 5)/1 + 1 = 10 -> (10, 10, 16)
        layers.Conv2D(16, kernel_size=(5, 5), activation='relu', name="Conv2"),
        layers.AveragePooling2D(pool_size=(2, 2), name="Pool2"), # Dim: (5, 5, 16)

        # --- PASSAGGIO ALLA CLASSIFICAZIONE ---
        # Flatten: 5 * 5 * 16 = 400 neuroni
        layers.Flatten(name="Flatten"),

        # Layer Denso 1: (400 ingressi + 1 bias) * 120 = 48,120 parametri
        layers.Dense(120, activation='relu', name="FC1"),

        # Layer Denso 2: (120 ingressi + 1 bias) * 84 = 10,164 parametri
        layers.Dense(84, activation='relu', name="FC2"),

        # Output Layer: (84 ingressi + 1 bias) * 10 classi = 850 parametri
        layers.Dense(10, activation='softmax', name="Output")
        
    ], name="LeNet-5_Modern")

    return model

# 1. Istanziamo il modello
model = build_lenet_modern()

# 2. Ispezione dettagliata
# Somma totale: 156 + 2416 + 48120 + 10164 + 850 = 61,706 parametri
model.summary()