import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# 1. Caricamento asincrono e divisione automatica (Punti chiave 1 e 3)
# Utilizziamo MNIST come esempio di dataset integrato
print("Caricamento dataset in corso...")
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Creazione di un oggetto Dataset per gestire la pipeline
# Usiamo il 20% del training per la validazione internamente durante il fit
# Il metodo shuffle e batch creano la pipeline asincrona
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

# 2. Costruzione del modello con Normalizzazione Integrata (Punto chiave 2)
model = models.Sequential([
    # Layer di riscalamento: trasforma i pixel da [0, 255] a [0, 1]
    layers.Rescaling(1./255, input_shape=(28, 28, 1)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 3. Compilazione e divisione automatica (validation_split)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Eseguiamo l'addestramento usando il validation_split automatico di Keras
# Nota: validation_split funziona direttamente su array numpy
history = model.fit(train_images, train_labels, 
                    epochs=3, 
                    validation_split=0.2, # Divisione automatica 80/20
                    batch_size=32)

print("\nAddestramento completato.")