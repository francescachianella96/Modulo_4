import tensorflow as tf
from tensorflow.keras import layers, models

def build_super_lenet(input_shape=(32, 32, 1)):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        # --- PRIMO BLOCCO: Sostituito 5x5 con due 3x3 ---
        # Conv 1a: ((3*3*1)+1)*6 = 60 param. Output: (30,30,6)
        layers.Conv2D(6, kernel_size=(3, 3), activation='relu', name="Conv1a"),
        # Conv 1b: ((3*3*6)+1)*6 = 330 param. Output: (28,28,6)
        layers.Conv2D(6, kernel_size=(3, 3), activation='relu', name="Conv1b"),
        # Sostituito Average con Max Pooling
        layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool1"), # Output: (14,14,6)

        # --- SECONDO BLOCCO: Sostituito 5x5 con due 3x3 ---
        # Conv 2a: ((3*3*6)+1)*16 = 880 param. Output: (12,12,16)
        layers.Conv2D(16, kernel_size=(3, 3), activation='relu', name="Conv2a"),
        # Conv 2b: ((3*3*16)+1)*16 = 2320 param. Output: (10,10,16)
        layers.Conv2D(16, kernel_size=(3, 3), activation='relu', name="Conv2b"),
        # Sostituito Average con Max Pooling
        layers.MaxPooling2D(pool_size=(2, 2), name="MaxPool2"), # Output: (5,5,16)

        # --- CLASSIFICAZIONE (Invariata) ---
        layers.Flatten(name="Flatten"),
        layers.Dense(120, activation='relu', name="FC1"),
        layers.Dense(84, activation='relu', name="FC2"),
        layers.Dense(10, activation='softmax', name="Output")
        
    ], name="Super-LeNet")

    return model

# Istanza e verifica
super_lenet = build_super_lenet()
super_lenet.summary()