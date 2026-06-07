import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import img_to_array
from keras.models import Model

# --- 1. CARICAMENTO DEL MODELLO PRE-ADDESTRATO ---
# Carichiamo VGG16 addestrata su ImageNet. 
# include_top=False: Escludiamo i layer Fully Connected finali (il classificatore).
# Ci fermiamo all'ultimo layer convolutivo per usare la rete come Feature Extractor.
base_model = VGG16(weights='imagenet', include_top=False)

# --- 2. PIPELINE DI ACQUISIZIONE E PRE-PROCESSING ---
def download_random_image(url="https://picsum.photos/224/224"):
    """
    Gestisce il recupero di dati binari via HTTP e la conversione in formato immagine.
    """
    print(f"Scaricando immagine da: {url}...")
    # Recupero dei byte grezzi dall'URL
    response = requests.get(url)
    
    # BytesIO crea un buffer binario in memoria che simula un file aperto.
    # PIL.Image.open legge il buffer e riconosce il formato (JPG/PNG).
    # .convert('RGB') è fondamentale: garantisce che l'immagine abbia 3 canali 
    # (alcune immagini web potrebbero essere in scala di grigi o avere il canale Alpha).
    img = Image.open(BytesIO(response.content)).convert('RGB')
    
    # Resize necessario: VGG16 si aspetta input di dimensione $224 \times 224$.
    img = img.resize((224, 224))
    return img

# Esecuzione del download
img = download_random_image()

# Conversione da oggetto PIL a array NumPy per manipolazione numerica.
x = img_to_array(img)

# Batch Dimension Expansion: Keras si aspetta tensori di forma (Batch, Altezza, Larghezza, Canali).
# Trasformiamo l'array da (224, 224, 3) a (1, 224, 224, 3).
x = np.expand_dims(x, axis=0)

# Preprocessing specifico ImageNet: 
# 1. Sottrae la media dei pixel calcolata sull'intero dataset ImageNet.
# 2. Converte i canali da RGB a BGR (standard storico di Caffe/VGG).
x = preprocess_input(x)

# --- 3. CREAZIONE DEL MODELLO DI ISPEZIONE (Functional API) ---
# Vogliamo estrarre l'output dei primi 8 layer (Blocchi convolutivi 1 e 2).
# base_model.layers[0] è l'InputLayer (non produce trasformazioni).
# Partiamo dall'indice 1 per visualizzare le prime attivazioni reali.
layer_outputs = [layer.output for layer in base_model.layers[1:9]]

# Definiamo un nuovo modello che ha lo stesso input del base_model 
# ma restituisce una LISTA di output (uno per ogni layer selezionato).
activation_model = Model(inputs=base_model.input, outputs=layer_outputs)

# Eseguiamo il passaggio in avanti (Forward Pass). 
# 'activations' conterrà 8 array NumPy con le mappe di caratteristiche (Feature Maps).
activations = activation_model.predict(x)

# --- 4. FUNZIONE DI VISUALIZZAZIONE SCIENTIFICA ---
def plot_layer_activations(layer_num, layer_name, activation_data, n_features=8):
    """
    Visualizza i primi n filtri (canali) estratti da un determinato layer.
    """
    # Selezioniamo il tensore delle attivazioni per il layer desiderato
    layer_activation = activation_data[layer_num]
    
    plt.figure(figsize=(16, 4))
    plt.suptitle(f"Layer {layer_num}: {layer_name} | Feature Maps", fontsize=16)
    
    for i in range(n_features):
        plt.subplot(1, n_features, i + 1)
        # Visualizziamo il canale i-esimo (indice dell'ultima dimensione).
        # cmap='viridis': Mappa di colore percettivamente uniforme, standard nel 2025 
        # per evidenziare le intensità di attivazione dei neuroni.
        plt.imshow(layer_activation[0, :, :, i], cmap='viridis')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualizzazione dell'input originale per confronto visivo
plt.figure(figsize=(4, 4))
plt.imshow(img)
plt.title("Immagine Originale (Input)")
plt.axis('off')
plt.show()

# Analizziamo il layer 'block1_conv1' (indice 0 nella nostra lista di output)
# Questo layer di solito estrae feature a basso livello: bordi, colori e linee.
plot_layer_activations(0, base_model.layers[1].name, activations)