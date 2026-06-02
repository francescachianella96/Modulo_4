import os
import datetime

# --- CONFIGURAZIONE AMBIENTE ---
# Disattiva i log di sistema non necessari di TensorFlow (info e warning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Disattiva le ottimizzazioni oneDNN per evitare differenze di precisione numerica trascurabili
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, Input

def main():
    # 1. PREPARAZIONE DATI
    print("Caricamento dati...")
    # Carichiamo il dataset MNIST (60.000 immagini di cifre scritte a mano)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # NORMALIZZAZIONE: Trasformiamo i valori dei pixel da [0, 255] a [0, 1]
    # Questo aiuta la rete neurale a convergere (imparare) più velocemente
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # 2. DEFINIZIONE DEL MODELLO (Architettura della Rete)
    # Creiamo un modello sequenziale (uno strato dopo l'altro)
    model = models.Sequential([
        # Definiamo l'input: immagini 28x28 pixel
        Input(shape=(28, 28), name="Input_Layer"),
        
        # Trasformiamo la matrice 28x28 in un vettore piatto di 784 elementi
        layers.Flatten(),
        
        # Strato denso (completamente connesso) con 128 neuroni
        # 'relu' è la funzione di attivazione standard per evitare la scomparsa del gradiente
        layers.Dense(128, activation='relu', name="Hidden_Layer_1"),
        
        # DROPOUT: Spegne casualmente il 20% dei neuroni durante il training
        # Serve a prevenire l'Overfitting (la rete non impara a memoria i dati)
        layers.Dropout(0.2, name="Regularization"),
        
        # Strato di output: 10 neuroni (uno per ogni cifra da 0 a 9)
        # 'softmax' trasforma l'output in probabilità (la somma di tutti i neuroni sarà 1)
        layers.Dense(10, activation='softmax', name="Output_Layer")
    ])

    # COMPILAZIONE: Definiamo come il modello deve imparare
    model.compile(
        optimizer='adam',                # Algoritmo di ottimizzazione (molto efficiente)
        loss='sparse_categorical_crossentropy', # Funzione di errore per classificazioni multi-classe
        metrics=['accuracy']             # Metrica per valutare la performance
    )

    # 3. CONFIGURAZIONE TENSORBOARD (Monitoraggio)
    # Creiamo una cartella specifica basata sul timestamp per ogni esecuzione
    log_dir = os.path.join(os.getcwd(), "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    # Il callback TensorBoard scriverà i log durante l'allenamento
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1, # Calcola la distribuzione dei pesi ad ogni epoca
        write_graph=True,  # Salva il grafico della struttura del modello
        write_images=True # Visualizza i pesi del modello come immagini in TensorBoard.
    )

    # 4. TRAINING (Allenamento)
    print(f"I log verranno salvati in: {log_dir}")
    model.fit(
        x_train, y_train, 
        epochs=20,                  # Numero di passaggi completi sui dati
        validation_data=(x_test, y_test), # Valuta la precisione su dati mai visti dopo ogni epoca
        callbacks=[tensorboard_callback]  # Attiva TensorBoard
    )

if __name__ == "__main__":
    main()

# per attivare la tensorboard: tensorboard --logdir="logs/fit"