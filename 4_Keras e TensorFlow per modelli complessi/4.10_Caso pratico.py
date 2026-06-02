import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import layers, models, applications, optimizers

# --- 1. CONFIGURAZIONE E SETUP AMBIENTE ---
# La riproducibilità è cruciale: fissare il seed garantisce che i pesi iniziali 
# e lo split del dataset siano identici ad ogni esecuzione.
tf.keras.utils.set_random_seed(123)

# AUTOTUNE permette al runtime di TensorFlow di regolare dinamicamente il valore di 
# parallelismo per l'estrazione dei dati in base alle risorse della CPU/GPU.
AUTOTUNE = tf.data.AUTOTUNE

# --- 2. DATA PIPELINE (tf.data) ---
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

def get_datasets(batch_size=32, img_size=(224, 224)):
    """
    Caricamento ottimizzato. L'uso di image_dataset_from_directory 
    astrae la gestione dei label basata sulla struttura delle cartelle.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=123, 
        image_size=img_size, batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=123, 
        image_size=img_size, batch_size=batch_size
    )
    
    # .cache(): mantiene i dati in memoria dopo la prima lettura, evitando colli di bottiglia dall'I/O del disco.
    # .shuffle(1000): garantisce che il modello non impari l'ordine dei dati (importante per la convergenza).
    # .prefetch(): prepara il batch successivo mentre la GPU sta elaborando il corrente (sovrapposizione CPU/GPU).
    return train_ds.cache().shuffle(1000).prefetch(AUTOTUNE), val_ds.cache().prefetch(AUTOTUNE)

train_ds, val_ds = get_datasets()

# --- 3. MODELLO CON HYPERPARAMETER TUNING (Bayesian Optimization) ---
def build_model(hp):
    """
    Costruisce il modello definendo lo spazio di ricerca. A differenza del Grid Search, 
    l'ottimizzazione Bayesiana impara dai tentativi precedenti per trovare il setup ottimale.
    """
    # Spazio di ricerca per Learning Rate (logaritmico per coprire diversi ordini di grandezza)
    hp_lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    # Dropout per prevenire l'overfitting (regolarizzazione)
    hp_dropout = hp.Float('dropout', 0.2, 0.5, step=0.1)
    
    # MobileNetV3Large: Architettura basata su Hard-Swish activation e Squeeze-and-Excitation.
    # Ideale per performance elevate con basso costo computazionale.
    base_model = applications.MobileNetV3Large(
        input_shape=(224, 224, 3), 
        include_top=False, # Rimuoviamo il "capo" originale per adattarlo alle nostre 5 classi
        weights='imagenet' # Utilizzo di pesi pre-addestrati su milioni di immagini (Transfer Learning)
    )
    
    # Freeze iniziale: non aggiorniamo i pesi del backbone per non distruggere 
    # le feature già apprese (bordi, forme) durante il tuning iniziale.
    base_model.trainable = False 
    
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        
        # Data Augmentation integrata nel grafo: avviene in tempo reale su GPU.
        # Rende il modello robusto a variazioni di angolazione e zoom.
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        base_model,
        # Riduce la dimensionalità da (7, 7, 960) a (960,) calcolando la media, 
        # riducendo drasticamente il numero di parametri rispetto a un layer Flatten.
        layers.GlobalAveragePooling2D(),
        layers.Dropout(hp_dropout),
        layers.Dense(5, activation='softmax') # Softmax per classificazione multi-classe mutua esclusiva
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=hp_lr),
        loss='sparse_categorical_crossentropy', # Usata perché le label sono interi, non one-hot encoded
        metrics=['accuracy']
    )
    return model



# Configurazione del Tuner: BayesianOptimization è più efficiente del RandomSearch
# poiché modella probabilisticamente la funzione obiettivo.
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=5, 
    directory='tuning_logs',
    project_name='flower_classification_2025'
)

print("--- Ricerca Bayesiana del miglior set di Iperparametri ---")
tuner.search(train_ds, epochs=5, validation_data=val_ds)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# --- 4. CALLBACKS E ADDESTRAMENTO FINALE ---

# Ricostruiamo il modello con i parametri vincenti trovati dal tuner
model = tuner.hypermodel.build(best_hps)

callbacks = [
    # Salvataggio nel nuovo formato .keras: più sicuro e leggero del vecchio .h5
    tf.keras.callbacks.ModelCheckpoint(
        'best_model_2025.keras', monitor='val_accuracy', save_best_only=True, verbose=1
    ),
    # EarlyStopping: interrompe l'addestramento se la loss non scende più, 
    # evitando spreco di risorse e overfitting.
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    ),
    # ReduceLROnPlateau: se il modello smette di imparare, "rallenta" per esplorare 
    # con più precisione i minimi della funzione di perdita.
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
    )
]

print(f"--- Addestramento finale con LR: {best_hps.get('learning_rate')} ---")
# Epochs=50 è un limite superiore; EarlyStopping probabilmente fermerà il processo prima.
history = model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=50, 
    callbacks=callbacks
)

# --- 5. VISUALIZZAZIONE PERFORMANCE ---
def plot_metrics(history):
    """Utility per analizzare il gap tra training e validation (segnale di overfitting)."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy', lw=2)
    plt.plot(val_acc, label='Validation Accuracy', lw=2)
    plt.title('Evoluzione Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss', lw=2)
    plt.plot(val_loss, label='Validation Loss', lw=2)
    plt.title('Evoluzione Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

plot_metrics(history)

# --- 6. TFLITE 8-BIT FULL INTEGER QUANTIZATION ---

# La quantizzazione trasforma i pesi da float32 (4 byte) a int8 (1 byte).
# Questo riduce la dimensione del modello di 4x e velocizza l'inferenza su dispositivi mobili.


def representative_data_gen():
    """
    Essenziale per la quantizzazione intera: fornisce un piccolo set di dati reali 
    per calibrare il range dinamico (min/max) delle attivazioni.
    """
    for input_value, _ in val_ds.take(20):
        yield [input_value]

print("--- Conversione in TFLite (8-bit Quantization) ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Forziamo l'uso di soli operatori interi. Se un'operazione non è supportata, il converter darà errore.
# Questo garantisce la compatibilità con acceleratori hardware come la Google Coral Edge TPU.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8 # Input come immagine 0-255
converter.inference_output_type = tf.uint8 # Output come probabilità quantizzata

tflite_model_quant = converter.convert()

with open('model_quantized_8bit.tflite', 'wb') as f:
    f.write(tflite_model_quant)

print("\nPipeline completata! Modelli pronti per il deployment.")