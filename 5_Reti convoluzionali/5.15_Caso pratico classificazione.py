#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PIPELINE DI CLASSIFICAZIONE BINARIA (CAT VS DOG) - EDIZIONE 2025
Fix applicato: Risoluzione errore READ_VARIABLE nel convertitore TFLite.
"""

import os, pathlib, random, zipfile, requests
from PIL import Image

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ------------------------------------------------------------------
# 0. IMPOSTAZIONI GLOBALI & REPRODUCIBILITÀ
# ------------------------------------------------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE   = tf.data.AUTOTUNE
IMG_SIZE   = (160, 160)
BATCH_SIZE = 32

# ------------------------------------------------------------------
# 1. ACQUISIZIONE DATASET
# ------------------------------------------------------------------
DATA_URL = (
    "https://download.microsoft.com/download/"
    "3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
)

BASE_DIR  = pathlib.Path.home() / ".keras" / "datasets"
BASE_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH   = BASE_DIR / "cats_and_dogs.zip"
EXTRACTED  = BASE_DIR / "PetImages"

if not ZIP_PATH.exists():
    print("[*] Scarico il dataset…")
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(DATA_URL, headers=headers)
    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)

if not EXTRACTED.exists():
    print("[*] Estrazione in corso…")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(BASE_DIR)

DATA_DIR = BASE_DIR / "PetImages"

# ------------------------------------------------------------------
# 2. SANITIZZAZIONE DATI (Data Cleaning migliorata)
# ------------------------------------------------------------------
def clean_images(directory: pathlib.Path):
    """
    Rimuove file corrotti o con header JPEG non standard che causano i warning 
    'Corrupt JPEG data' e bloccano il training/conversione.
    """
    bad_files = 0
    for cls in ("Cat", "Dog"):
        folder = directory / cls
        for fname in os.listdir(folder):
            fpath = folder / fname
            if fpath.is_dir(): continue
            try:
                # La decodifica PIL è spesso più sensibile di tf.io alle corruzioni di byte
                with Image.open(fpath) as img:
                    img.verify() 
                # Ulteriore controllo via TF
                img_bytes = tf.io.read_file(str(fpath))
                _ = tf.image.decode_jpeg(img_bytes, channels=3)
            except Exception:
                bad_files += 1
                fpath.unlink()
    print(f"[+] {bad_files} file corrotti rimossi.")

clean_images(DATA_DIR)

# ------------------------------------------------------------------
# 3. PIPELINE DI INPUT (tf.data)
# ------------------------------------------------------------------
def get_datasets():
    raw_train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    raw_val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = raw_train_ds.class_names

    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.05),
    ])

    def prepare(ds, augment=False):
        # Normalizzazione a [0, 1]
        ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)
        if augment:
            ds = ds.map(
                lambda x, y: (data_aug(x, training=True), y),
                num_parallel_calls=AUTOTUNE,
            )
        return ds.prefetch(AUTOTUNE)

    train_ds = prepare(raw_train_ds, augment=True)
    val_ds   = prepare(raw_val_ds, augment=False)

    return train_ds, val_ds, class_names

train_ds, val_ds, class_names = get_datasets()

# ------------------------------------------------------------------
# 4. ARCHITETTURA DEL MODELLO (CNN)
# ------------------------------------------------------------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5), 
        layers.Dense(1, activation="sigmoid"),
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

model = build_model()
print("\n=== Architettura del modello ===")
model.summary()

# ------------------------------------------------------------------
# 5. STRATEGIE DI ADDESTRAMENTO
# ------------------------------------------------------------------
EPOCHS = 1 # Impostato a 1 per test rapido, aumentare in produzione

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
)

# ------------------------------------------------------------------
# 6. DIAGNOSTICA POST-TRAINING
# ------------------------------------------------------------------
def per_class_accuracy(model, val_ds):
    y_true, y_pred = [], []
    for x_batch, y_batch in val_ds:
        preds = model.predict(x_batch, verbose=0).flatten()
        y_true.extend(y_batch.numpy())
        y_pred.extend((preds > 0.5).astype(int))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n--- Accuracy per classe ---")
    for idx, name in enumerate(class_names):
        mask = (y_true == idx)
        acc = np.mean(y_pred[mask] == y_true[mask])
        print(f"{name:>5}: {acc:.2%}")

per_class_accuracy(model, val_ds)

# ------------------------------------------------------------------
# 7. SALVATAGGIO E QUANTIZZAZIONE (Fix RuntimeError)
# ------------------------------------------------------------------
MODEL_DIR = "saved_model_pets"
tf.saved_model.save(model, MODEL_DIR)
print(f"\n[+] Modello salvato in {MODEL_DIR}")

def representative_data_gen():
    """
    Generatore per la calibrazione INT8. 
    Prendiamo i dati direttamente dal dataset già normalizzato.
    """
    # Prendiamo 100 campioni per la calibrazione
    for images, _ in train_ds.take(100):
        # I dati in train_ds sono già in batch di 32. 
        # TFLite richiede un sample alla volta o il batch intero? 
        # Iteriamo sul batch per sicurezza.
        for i in range(images.shape[0]):
            img = images[i:i+1] # Mantiene la forma (1, 160, 160, 3)
            yield [img.numpy()]

# FIX: Usiamo 'from_keras_model' invece di 'from_saved_model'.
# Questo risolve l'errore READ_VARIABLE perché il convertitore ha accesso
# diretto all'istanza Keras e alle sue variabili tracciate correttamente.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Queste righe forzano il modello ad accettare INT8 in ingresso e uscita
# Utile per Edge TPU o microcontrollori.
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

try:
    tflite_quantized = converter.convert()
    with open("model_pets_quantized.tflite", "wb") as f:
        f.write(tflite_quantized)
    print("[+] Modello quantizzato TFLite scritto con successo!")
except Exception as e:
    print(f"[-] Errore durante la conversione: {e}")