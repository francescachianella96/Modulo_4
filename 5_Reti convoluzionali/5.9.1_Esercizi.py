import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import numpy as np

# --- 1. PREPARAZIONE DATI ---
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

# --- 2. ARCHITETTURA CON TERZO LAYER (Super-MNIST) ---
def build_model():
    inputs = Input(shape=(28, 28, 1), name="Input_Layer")
    
    # Primo blocco: 28x28 -> 26x26 -> 13x13
    x = layers.Conv2D(32, (3, 3), activation='relu', name='Conv1')(inputs)
    x = layers.MaxPooling2D((2, 2), name='Pool1')(x)
    
    # Secondo blocco: 13x13 -> 11x11 -> 5x5
    x = layers.Conv2D(64, (3, 3), activation='relu', name='Conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='Pool2')(x)
    
    # TERZO LAYER AGGIUNTO: 5x5 -> 3x3
    # Calcolo parametri: ((3*3 * 64 in) + 1) * 64 out = 36.928
    x = layers.Conv2D(64, (3, 3), activation='relu', name='Conv3')(x)
    
    # Testa del modello: Il volume prima di Flatten è ora solo 3x3x64
    x = layers.Flatten()(x) 
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="MNIST_Deep_2025")

model = build_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary() 

# Riflessione: La dimensione spaziale è crollata a 3x3. 
# Questo riduce drasticamente i parametri del layer Dense (Flatten passa da 1600 a 576 neuroni).

# Training rapido
model.fit(train_images, train_labels, epochs=3, batch_size=64, verbose=1)

# --- 3. ESTRAZIONE ATTIVAZIONI TERZO LIVELLO ---
img = test_images[0:1] 
# Puntiamo all'output del NUOVO layer Conv3
activation_model = models.Model(inputs=model.input, outputs=model.get_layer('Conv2').output)
activations = activation_model.predict(img, verbose=0) # Forma: (1, 3, 3, 64)

# --- 4. VISUALIZZAZIONE ---
plt.figure(figsize=(16, 8))
for i in range(8):
    plt.subplot(2, 4, i+1)
    # Nota: L'immagine apparirà molto "pixelata" (3x3 pixel) perché siamo molto in profondità
    plt.imshow(activations[0, :, :, i], cmap='magma')
    plt.axis('off')
    plt.title(f"Feature Map Conv3_{i}")

plt.suptitle("Astrazione Profonda: Cosa 'vede' il terzo layer (3x3)", fontsize=16)
plt.show()