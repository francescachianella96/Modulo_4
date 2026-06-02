import os
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers

# --- 1. SETUP ---
tf.keras.utils.set_random_seed(123)
AUTOTUNE = tf.data.AUTOTUNE

# --- 2. DATA PIPELINE ---
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

def get_datasets(batch_size=32, img_size=(224, 224)):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=123, 
        image_size=img_size, batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=123, 
        image_size=img_size, batch_size=batch_size
    )
    return train_ds.cache().prefetch(AUTOTUNE), val_ds.cache().prefetch(AUTOTUNE)

train_ds, val_ds = get_datasets()

# --- 3. DEFINIZIONE ARCHITETTURE ---

def create_model(robust=False):
    base_model = applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False 
    
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        base_model,
        layers.GlobalAveragePooling2D(),
    ])
    
    if robust:
        # Testa Robusta richiesta
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Dense(5, activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 4. ADDESTRAMENTO COMPARATIVO (2 EPOCHE) ---

print("\n--- Training Modello BASE ---")
model_base = create_model(robust=False)
history_base = model_base.fit(train_ds, validation_data=val_ds, epochs=2)

print("\n--- Training Modello ROBUSTO ---")
model_robust = create_model(robust=True)
history_robust = model_robust.fit(train_ds, validation_data=val_ds, epochs=2)

# --- 5. GRAFICO DI CONFRONTO ---

def plot_comparison(h1, h2):
    epochs_range = range(1, 3)
    plt.figure(figsize=(12, 5))

    # Accuratezza
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, h1.history['val_accuracy'], 'o-', label='Base (Val Acc)')
    plt.plot(epochs_range, h2.history['val_accuracy'], 's-', label='Robusto (Val Acc)')
    plt.title('Confronto Accuratezza Validazione')
    plt.xlabel('Epoca')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Perdita
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, h1.history['val_loss'], 'o-', label='Base (Val Loss)')
    plt.plot(epochs_range, h2.history['val_loss'], 's-', label='Robusto (Val Loss)')
    plt.title('Confronto Perdita Validazione')
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_comparison(history_base, history_robust)