import keras
from keras import layers, models 
import numpy as np

# --- PREPARAZIONE DATI (Fashion MNIST) ---
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0  # Normalizzazione
x_train = np.expand_dims(x_train, -1)        # Da (28,28) a (28,28,1)

def build_comparison_cnn(use_bn=True):
    model = models.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))

    # --- BLOCCO 1 ---
    # Se use_bn è True, togliamo il bias per efficienza
    model.add(layers.Conv2D(32, (3, 3), padding='same', use_bias=not use_bn))
    if use_bn: model.add(layers.BatchNormalization()) 
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # --- BLOCCO 2 ---
    model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=not use_bn))
    if use_bn: model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.SpatialDropout2D(0.3)) 
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5)) 
    model.add(layers.Dense(10, activation='softmax'))
    return model

# --- CONFRONTO ---
for name, use_bn in [("Modello A (No BN)", False), ("Modello B (Con BN)", True)]:
    print(f"\n--- ADDDESTRAMENTO {name} ---")
    model = build_comparison_cnn(use_bn=use_bn)
    # Impostiamo un Learning Rate elevato (0.01) come richiesto
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
    model.summary()