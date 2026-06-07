import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# --- 1. DEFINIZIONE DELLA PIPELINE (Punto chiave 2) ---
# Creiamo un modello dedicato solo all'augmentation
data_augmentation = models.Sequential([
  layers.RandomFlip("horizontal"),        # Specchio orizzontale (Flip)
  layers.RandomRotation(0.2),             # Rotazione casuale (+/- 20%)
  layers.RandomZoom(0.1),                 # Zoom casuale (+/- 10%)
])

# --- 2. CARICAMENTO E PREPARAZIONE (Punto chiave 1) ---
# Usiamo CIFAR10 come base di test
(x_train, _), _ = tf.keras.datasets.cifar10.load_data()
image = x_train[0] # Prendiamo una singola immagine (es. un aereo)
image = np.expand_dims(image, 0) # Aggiungiamo la dimensione batch

# --- 3. VISUALIZZAZIONE (Punto chiave 3) ---
plt.figure(figsize=(10, 10))
for i in range(9):
    # Applichiamo l'augmentation sulla stessa immagine
    augmented_image = data_augmentation(image)
    
    plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0].numpy().astype("uint8"))
    plt.axis("off")
    plt.title(f"Variante {i+1}")

plt.tight_layout()
plt.show()