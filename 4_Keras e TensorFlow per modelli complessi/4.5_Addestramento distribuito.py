import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# 1. DEFINIZIONE DELLA STRATEGIA (Punto chiave 1 e 3)
# Rileva automaticamente tutte le GPU visibili
strategy = tf.distribute.MirroredStrategy()
print(f"Numero di dispositivi arruolati: {strategy.num_replicas_in_sync}")

# 2. SCALARE IL BATCH SIZE
# Se vogliamo che ogni GPU lavori con 32 campioni (local batch),
# il batch globale deve essere 32 * numero_di_gpu.
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

# 3. PREPARAZIONE DEI DATI (tf.data raccomandato)
(train_images, train_labels), _ = datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255

# Creiamo un dataset ottimizzato
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(10000).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 4. COSTRUZIONE DEL MODELLO DENTRO LO SCOPE (Punto chiave 3)
with strategy.scope():
    # Tutto ciò che viene creato qui sarà replicato su tutte le GPU
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

# 5. ADDESTRAMENTO
# Keras gestirà automaticamente la distribuzione dei batch tra le GPU
model.fit(train_dataset, epochs=5)