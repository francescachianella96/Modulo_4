import os

# Configurazione del Backend: Fondamentale nel 2026. Keras 3 è "multi-backend".
# Impostando "torch", chiediamo a Keras di tradurre i suoi layer in moduli PyTorch nativi.
# Va fatto PRIMA di importare keras perché la scelta del motore di calcolo è immutabile dopo il caricamento.
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

# Scarichiamo il dataset: 50.000 immagini di training e 10.000 di test (32x32 pixel a colori).
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Funzione di filtraggio binario: essenziale per restringere il dominio del problema.
def filter_binary(x, y, c1, c2):
    # Creiamo una 'maschera booleana': True solo dove la classe è c1 o c2.
    mask = (y == c1) | (y == c2)
    
    # Applichiamo la maschera per estrarre solo le immagini desiderate.
    x_filtered = x[mask.flatten()]
    
    # Trasformiamo le etichette in 0 (per gatti, c1) e 1 (per cani, c2).
    # .astype(int) converte i Booleani risultanti in numeri interi (0 e 1).
    y_filtered = (y[mask.flatten()] == c2).astype(int) 
    return x_filtered, y_filtered

# Applichiamo il filtro: Classe 3 = Cat, Classe 5 = Dog.
x_train, y_train = filter_binary(x_train, y_train, 3, 5)
x_test, y_test = filter_binary(x_test, y_test, 3, 5)


# MobileNetV2: Un modello leggero ed efficiente basato su 'Inverted Residual Blocks'.
# include_top=False: Rimuoviamo il layer finale di ImageNet (che predice 1000 classi).
# weights="imagenet": Carichiamo i pesi già ottimizzati per riconoscere forme e texture universali.
base_model = keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False, 
    weights="imagenet"
)

# CONGELAMENTO (Freezing): Impostiamo trainable=False.
# Teoria: Vogliamo che i filtri visivi della base (che sanno già distinguere forme e colori) 
# non vengano modificati durante la prima fase, altrimenti i pesi casuali del nuovo "capo" 
# invierebbero gradienti forti che distruggerebbero le conoscenze pre-esistenti.
base_model.trainable = False

# --- Pipeline Funzionale Keras ---
# Input: Definiamo la forma d'ingresso delle immagini CIFAR10 (32x32x3).
inputs = keras.Input(shape=(32, 32, 3)) 

# Resizing: MobileNetV2 è stata addestrata su immagini grandi. 
# Fare upscaling a 160x160 migliora drasticamente la qualità delle feature estratte.
x = keras.layers.Resizing(160, 160)(inputs) 

# Preprocess_input: Normalizza i pixel. MobileNet si aspetta valori nel range [-1, 1].
x = keras.applications.mobilenet_v2.preprocess_input(x) 

# Backbone: Passiamo l'immagine nel modello base. 
# training=False è critico: mantiene i layer di Batch Normalization in modalità inferenza.
x = base_model(x, training=False) 

# GlobalAveragePooling2D: Riduce la mappa di feature (es. 5x5x1280) a un vettore singolo (1280).
# Più moderno e con meno parametri rispetto a un classico layer Flatten.
x = keras.layers.GlobalAveragePooling2D()(x) 

# Dropout: Spegne casualmente il 20% dei neuroni per forzare la rete a non dipendere da singoli 
# segnali, riducendo l'overfitting (la memorizzazione a memoria dei dati).
x = keras.layers.Dropout(0.2)(x) 

# Output: Sigmoid trasforma l'output in una probabilità tra 0 e 1.
# Formula: $S(x) = \frac{1}{1 + e^{-x}}$
outputs = keras.layers.Dense(1, activation="sigmoid")(x) 

model = keras.Model(inputs, outputs)

# Compilazione Fase 1: Usiamo Adam con un Learning Rate standard (0.001).
# loss="binary_crossentropy": La funzione di costo ideale per probabilità binarie.
# Formula: $L = -\frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])


# FASE 1: Addestriamo solo la "testa" del modello (i layer densi aggiunti da noi).
print("[INFO] Fase 1: Feature Extraction...")
model.fit(x_train, y_train, epochs=3, validation_split=0.2, batch_size=32)

# FASE 2: FINE-TUNING
# Teoria: Ora sblocchiamo la base per permettere al modello di specializzare i suoi filtri 
# visivi profondi sulle caratteristiche specifiche di cani e gatti di CIFAR10.
base_model.trainable = True

# Strategia: Congeliamo i primi 100 layer (che vedono linee e bordi generici) e 
# sblocchiamo solo i layer finali (che vedono strutture complesse come orecchie o musi).
for layer in base_model.layers[:100]:
    layer.trainable = False

# CRITICO: Usiamo un Learning Rate minuscolo (1e-5). 
# Un passo troppo grande distruggerebbe i pesi di ImageNet (Catastrophic Forgetting).
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="binary_crossentropy", metrics=["accuracy"])

print("[INFO] Fase 2: Fine-Tuning...")
model.fit(x_train, y_train, epochs=3, validation_split=0.2, batch_size=32)

# Salvataggio: Il formato .keras è un pacchetto ZIP che contiene tutto il necessario per il deploy.
model.save("classifier_immagini_2026.keras")


# Predict: Otteniamo le probabilità (es. 0.85 invece di un secco "Cane").
y_pred_probs = model.predict(x_test).ravel()

# ROC Curve: Analizza il trade-off tra Veri Positivi e Falsi Positivi.
# L'AUC (Area Under Curve) indica quanto il modello è bravo a separare le due classi (1.0 è perfetto).
fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve: Più indicata se le classi fossero sbilanciate.
precision, recall, _ = precision_recall_curve(y_test, y_pred_probs)

# Visualizzazione con Matplotlib
plt.figure(figsize=(12, 5))

# Sottografo ROC
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # Linea del caso (50/50)
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend()

# Sottografo PR
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', label='PR curve')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall (Capacità di trovare tutti i positivi)')
plt.ylabel('Precision (Capacità di non sbagliare i positivi)')
plt.legend()

plt.show()