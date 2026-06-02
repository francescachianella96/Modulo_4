import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# =============================================================================
# 1. CARICAMENTO ASINCRONO E PREPARAZIONE DATI
# =============================================================================
# Carichiamo CIFAR-10: immagini 32x32 a colori (3 canali RGB)
print("Caricamento dataset CIFAR-10 in corso...")
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# --- CREAZIONE PIPELINE TF.DATA ---
# Anche se useremo il validation_split automatico di Keras sotto, 
# ecco come si configura una pipeline asincrona completa:
# 1. from_tensor_slices: "affetta" i dati in coppie (immagine, etichetta)
# 2. shuffle(10000): mischia i dati con un buffer di 10.000 elementi
# 3. batch(64): raggruppa le immagini in lotti da 64
# 4. prefetch: prepara il batch successivo mentre la GPU lavora (massima efficienza)
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)

# =============================================================================
# 2. COSTRUZIONE MODELLO CON NORMALIZZAZIONE [-1, 1]
# =============================================================================
model = models.Sequential([
    # Layer di riscalamento per mappare [0, 255] nell'intervallo [-1, 1]
    # Formula: (input * 1/127.5) - 1
    # Esempio: 255 * (1/127.5) - 1 = 2 - 1 = 1
    # Esempio: 0 * (1/127.5) - 1 = -1
    layers.Rescaling(1./127.5, offset=-1, input_shape=(32, 32, 3)),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# =============================================================================
# 3. COMPILAZIONE E ADDESTRAMENTO (Gestione automatica TF)
# =============================================================================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Facciamo gestire la validazione direttamente a TF/Keras:
# Passando gli array NumPy, Keras applicherà internamente lo shuffle e il batching
# permettendoci di usare 'validation_split' per isolare il 15% dei dati.
print("\nInizio addestramento con gestione automatica della validazione...")
history = model.fit(
    train_images, 
    train_labels, 
    epochs=5, 
    batch_size=64, 
    validation_split=0.15, # TF isola automaticamente il 15% per la validazione
    shuffle=True           # TF mischia i dati ad ogni epoca
)

# Verifica finale sui dati di test
print("\nValutazione sul Test Set:")
model.evaluate(test_images, test_labels)