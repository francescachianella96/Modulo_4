import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# 1. DEFINIZIONE DELLA STRATEGIA
# MirroredStrategy gestisce la replicazione sincrona su tutte le GPU del nodo
strategy = tf.distribute.MirroredStrategy()

# Conferma del numero di repliche attive (Richiesta esercizio)
print(f"--- VERIFICA HARDWARE: Numero di repliche attive: {strategy.num_replicas_in_sync} ---")

# 2. CALCOLO DINAMICO DEL BATCH SIZE
# Per processare 128 immagini per singola GPU, dobbiamo moltiplicare 
# questo valore per il numero totale di GPU (repliche).
BATCH_SIZE_PER_REPLICA = 128
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print(f"Batch Size Locale: {BATCH_SIZE_PER_REPLICA}")
print(f"Batch Size Globale (totale): {GLOBAL_BATCH_SIZE}")

# 3. PREPARAZIONE DEI DATI
(train_images, train_labels), _ = datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255

# Pipeline ottimizzata con tf.data
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
# Lo shuffle deve essere sufficientemente grande da garantire varietà statistica
train_dataset = train_dataset.shuffle(10000).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 4. COSTRUZIONE DEL MODELLO NELLO SCOPE DELLA STRATEGIA
# Definire il modello dentro 'strategy.scope()' assicura che i pesi
# siano specchiati su tutte le GPU.
with strategy.scope():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# 5. ESECUZIONE DELL'ADDESTRAMENTO
# Il metodo fit distribuirà automaticamente i campioni del GLOBAL_BATCH_SIZE
# in parti uguali (128 ciascuna) tra le GPU.
model.fit(train_dataset, epochs=5)