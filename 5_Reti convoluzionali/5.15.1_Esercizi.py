import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, datasets

# ------------------------------------------------------------------
# 0. CONFIGURAZIONE E REPRODUCIBILITÀ
# ------------------------------------------------------------------
SEED = 42
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

# Mappatura classi CIFAR-10: Bird(2), Cat(3), Dog(5)
TARGET_CLASSES = [2, 3, 5]
IMG_SIZE = (32, 32) # Dimensione nativa CIFAR-10
BATCH_SIZE = 64

# ------------------------------------------------------------------
# 1. CARICAMENTO E FILTRAGGIO DATI
# ------------------------------------------------------------------
print("[*] Caricamento e filtraggio CIFAR-10...")
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = datasets.cifar10.load_data()

def filter_classes(x, y, target_classes):
    # Crea una maschera booleana per le classi desiderate
    mask = np.isin(y, target_classes).flatten()
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    # Rimappatura label: {2, 3, 5} -> {0, 1, 2}
    # Indispensabile per avere un output layer coerente
    mapping = {old: new for new, old in enumerate(target_classes)}
    y_remapped = np.array([mapping[val[0]] for val in y_filtered])
    
    return x_filtered, y_remapped

x_train, y_train = filter_classes(x_train_raw, y_train_raw, TARGET_CLASSES)
x_test, y_test = filter_classes(x_test_raw, y_test_raw, TARGET_CLASSES)

print(f"[+] Dataset filtrato: {len(x_train)} campioni training, {len(x_test)} test.")

# ------------------------------------------------------------------
# 2. PIPELINE TF.DATA & DATA AUGMENTATION
# ------------------------------------------------------------------
# Definiamo l'augmentation come richiesto (incluso RandomContrast)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomContrast(0.2), # Richiesto: contrasto casuale del 20%
])

def prepare_ds(x, y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(1000) if training else ds
    ds = ds.batch(BATCH_SIZE)
    
    # Normalizzazione [0, 1] e Augmentation
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    
    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    
    return ds.prefetch(AUTOTUNE)

train_ds = prepare_ds(x_train, y_train, training=True)
test_ds  = prepare_ds(x_test, y_test, training=False)

# ------------------------------------------------------------------
# 3. ARCHITETTURA DEL MODELLO
# ------------------------------------------------------------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax') # 3 classi: Bird, Cat, Dog
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()

# Training rapido per dimostrazione
model.fit(train_ds, validation_data=test_ds, epochs=10)

# ------------------------------------------------------------------
# 4. EXPORT TFLITE CON QUANTIZZAZIONE FLOAT16
# ------------------------------------------------------------------
print("\n[*] Conversione in TFLite (Float16 Quantization)...")

# Utilizziamo la conversione diretta da modello Keras (best practice 2025)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Abilitiamo l'ottimizzazione e forziamo Float16
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_fp16_model = converter.convert()

# Salvataggio
with open("cifar10_specialist_fp16.tflite", "wb") as f:
    f.write(tflite_fp16_model)

print("[+] Modello Float16 salvato con successo!")