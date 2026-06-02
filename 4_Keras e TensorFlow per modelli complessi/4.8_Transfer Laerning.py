import tensorflow as tf
from tensorflow.keras import layers, models, applications

# 1. CARICAMENTO DEL MODELLO BASE (Punto chiave 3)
# include_top=False rimuove i layer densi finali (il classificatore a 1000 classi)
vgg_base = applications.VGG16(weights='imagenet', 
                             include_top=False, 
                             input_shape=(224, 224, 3))

# 2. CONGELAMENTO DEI PESI (Punto chiave 2)
# Impediamo all'ottimizzatore di modificare i pesi già appresi su ImageNet
vgg_base.trainable = False

# 3. COSTRUZIONE DEL MODELLO COMPLESSIVO
model = models.Sequential([
    vgg_base,                    # La base convoluzionale "intelligente"
    layers.GlobalAveragePooling2D(), # Trasforma le mappe 2D in un vettore 1D
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),         # Regolarizzazione per evitare overfitting
    layers.Dense(10, activation='softmax') # Esempio: 10 nuove categorie
])

# 4. COMPILAZIONE
# Usiamo Adam con il learning rate standard perché la base è congelata
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Ispezione dei parametri
model.summary()

print(f"\nLayer del modello base congelati: {len(vgg_base.layers)}")