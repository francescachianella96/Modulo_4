import os
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.applications import ResNet50

# 1. CONFIGURAZIONE BACKEND
os.environ["KERAS_BACKEND"] = "tensorflow"

# --- 2. ARCHITETTURA DEL MODELLO ---

# Carichiamo ResNet50 pre-addestrata su ImageNet senza il classificatore finale
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# BLOCCAGGIO INIZIALE: Congeliamo tutta la base convoluzionale
base_model.trainable = False

# Costruzione della TESTA DI CLASSIFICAZIONE (Head)
inputs = layers.Input(shape=(224, 224, 3))
# Pre-processing specifico per ResNet
x = keras.applications.resnet50.preprocess_input(inputs)
x = base_model(x, training=False) # training=False assicura che BatchNormalization resti in inference mode
x = layers.GlobalAveragePooling2D()(x) # Converte le feature map in un vettore
x = layers.Dropout(0.5)(x)             # Regolarizzazione
outputs = layers.Dense(5, activation='softmax')(x) # 5 classi per il dataset fiori

model = models.Model(inputs, outputs, name="ResNet50_Flower_Model")

# Compilazione iniziale (Warm-up della testa)
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# --- 3. SBLOCCO SELETTIVO (FINE-TUNING) ---

# Sblocchiamo la base convoluzionale
base_model.trainable = True

# CODICE PER SBLOCCARE ESATTAMENTE GLI ULTIMI 15 LAYER
# Iteriamo sui layer e congeliamo tutti tranne gli ultimi 15
for layer in base_model.layers[:-15]:
    layer.trainable = False

# --- 4. RICOMPILAZIONE ---

# Ricompilazione obbligatoria con learning rate ridotto
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print(f"Modello pronto. Layer totali in base_model: {len(base_model.layers)}")
print(f"Layer sbloccati per il fine-tuning: 15")

# --- SPIEGAZIONE LEARNING RATE ---
"""
Perché usiamo un learning rate così piccolo (1e-5)?
Quando eseguiamo il fine-tuning, stiamo lavorando su un modello che possiede già pesi 
estremamente ottimizzati per il riconoscimento di forme visive (grazie a ImageNet). 

1. Prevenzione della distruzione delle feature: Un learning rate elevato causerebbe 
   aggiornamenti dei pesi troppo aggressivi, "distruggendo" le conoscenze pregresse 
   della rete e portando al fenomeno del 'catastrophic forgetting'.
2. Adattamento delicato: Un valore piccolo permette ai pesi degli ultimi layer 
   convoluzionali di adattarsi dolcemente alle caratteristiche specifiche del nuovo 
   dataset (i fiori) senza perdere le capacità generali di estrazione delle feature.
"""