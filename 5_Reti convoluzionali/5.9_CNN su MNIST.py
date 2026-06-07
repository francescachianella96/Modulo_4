import os

# 1. CONFIGURAZIONE BACKEND (Best Practice 2026)
# Keras 3 permette di scegliere il motore di calcolo (backend). 
# Impostiamo "torch" prima di importare keras per sfruttare l'ecosistema PyTorch.
# Teoria: L'agnosticismo del backend permette di addestrare su un framework e distribuire su un altro.
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models, ops
import numpy as np
import tensorflow as tf # Utilizzato esclusivamente per la pipeline tf.data (standard industriale per prefetch)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 2. PIPELINE DATI OTTIMIZZATA (Prefetch & Cache)
# ==========================================

def prepare_dataset(x, y, training=True):
    """
    Crea una pipeline di dati performante.
    Teoria: Il collo di bottiglia nel DL è spesso il caricamento dati (CPU) rispetto al calcolo (GPU).
    """
    # Creiamo un oggetto Dataset dai tensori NumPy
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    
    # [TEORIA] Normalizzazione: Portiamo i pixel da [0, 255] a [0, 1].
    # Input piccoli e centrati evitano che i gradienti esplodano durante le prime epoche.
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    # [CACHE] Memorizza i dati in RAM dopo la prima lettura.
    # Evita di ripetere le operazioni di decodifica/preprocessing ad ogni epoca.
    dataset = dataset.cache()
    
    if training:
        # [SHUFFLE] Fondamentale per la convergenza: evita che il modello impari l'ordine dei dati.
        dataset = dataset.shuffle(buffer_size=5000)
    
    # Batching: raggruppiamo i dati.
    dataset = dataset.batch(64)
    
    # [PREFETCH] La "Killer Feature" per le performance.
    # Mentre la GPU calcola il gradiente del batch corrente (N), 
    # la CPU prepara già il batch successivo (N+1) in background.
    # AUTOTUNE decide dinamicamente quanti batch precaricare in base alle risorse.
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

# Caricamento dati CIFAR-10
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = keras.datasets.cifar10.load_data()

# Preparazione pipeline
train_ds = prepare_dataset(x_train_raw, y_train_raw, training=True)
test_ds = prepare_dataset(x_test_raw, y_test_raw, training=False)

# ==========================================
# 3. ARCHITETTURA: RESNET CUSTOM (API Funzionale)
# ==========================================

def residual_block(x, filters, stride=1):
    """
    Blocco Residuale (ResNet). 
    Teoria: Introduce le 'skip connections' per risolvere il problema della scomparsa del gradiente.
    Permette ai gradienti di fluire direttamente attraverso la rete durante la backpropagation.
    """
    shortcut = x
    
    # Percorso principale - Primo blocco: Conv -> BN -> Activation
    # La Batch Normalization stabilizza l'apprendimento normalizzando le attivazioni medie.
    x = layers.Conv2D(filters, 3, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Secondo blocco: Conv -> BN
    x = layers.Conv2D(filters, 3, strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # [MATCHING DIMENSIONI] Se cambiamo risoluzione (stride > 1), dobbiamo adattare anche lo shortcut
    # in modo che la somma element-wise sia matematicamente possibile.
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
        
    # [TEORIA] H(x) = F(x) + x. Aggiungiamo l'input originale all'output trasformato.
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def build_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    
    # Layer iniziale di estrazione feature (Stem)
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Stack di blocchi residuali con aumento progressivo dei filtri
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2) # Dimezza risoluzione (16x16)
    x = residual_block(x, 256, stride=2) # Dimezza risoluzione (8x8)
    
    # [GLOBAL AVERAGE POOLING] Best practice 2026 al posto del Flatten().
    # Riduce il numero di parametri e rende il modello più robusto alle traslazioni spaziali.
    x = layers.GlobalAveragePooling2D()(x)
    
    # Output layer con Softmax per classificazione multi-classe.
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return models.Model(inputs, outputs, name="ResNet_CIFAR10_2026")

model = build_model()
model.summary()

# ==========================================
# 4. COMPILAZIONE E TRAINING
# ==========================================

# [ADAMW] Optimizer standard nel 2026: Adam con Weight Decay integrato per una migliore regolarizzazione.
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# [CALLBACKS] Automazione del training
callbacks = [
    # Riduce il LR se la loss smette di migliorare (ottimizzazione fine)
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
    # Ferma il training per evitare l'overfitting
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

print("\n[INFO] Avvio addestramento con backend PyTorch...")
history = model.fit(
    train_ds, 
    validation_data=test_ds, 
    epochs=10, 
    callbacks=callbacks
)

# ==========================================
# 5. SALVATAGGIO MODELLO (Formato 2026)
# ==========================================

# Il formato .keras è lo standard V3: contiene pesi, architettura e stato dell'ottimizzatore.
# È un file compresso e platform-independent.
model.save("cifar10_model_v2026.keras")
print(f"\n[INFO] Modello salvato con successo: cifar10_model_v2026.keras")

# ==========================================
# 6. VALUTAZIONE E ANALISI
# ==========================================

# Generazione predizioni
y_pred_all = []
y_true_all = []

for x_batch, y_batch in test_ds:
    preds = model.predict(x_batch, verbose=0)
    y_pred_all.extend(np.argmax(preds, axis=1))
    y_true_all.extend(y_batch.numpy())

# Report di classificazione
target_names = ['Aereo', 'Auto', 'Uccello', 'Gatto', 'Cervo', 'Cane', 'Rana', 'Cavallo', 'Nave', 'Camion']
print("\n--- PERFORMANCE SUL TEST SET ---")
print(classification_report(y_true_all, y_pred_all, target_names=target_names))

# Matrice di Confusione Visiva
cm = confusion_matrix(y_true_all, y_pred_all)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Analisi Errori (Confusion Matrix)")
plt.colorbar()
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.show()