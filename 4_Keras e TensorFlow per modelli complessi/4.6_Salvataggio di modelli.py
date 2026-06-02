import tensorflow as tf
import keras 
import numpy as np
import os

def create_model():
    """
    Definisce un modello Sequential utilizzando la sintassi moderna.
    L'uso di keras.Input è fondamentale per inizializzare correttamente i pesi 
    prima del salvataggio.
    """
    model = keras.Sequential([
        keras.Input(shape=(10,)), # Specifica esplicitamente la forma dell'input
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilazione: definisce l'ottimizzatore e la funzione di perdita
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Inizializziamo il modello
model = create_model()

# --- 1. SALVATAGGIO NATIVO (.keras) ---
# Questa è la "best practice" assoluta nel 2025. 
# Salva tutto (architettura, pesi, stato dell'ottimizzatore) in un unico file compresso.
model.save('my_model.keras') 

# --- 2. ESPORTAZIONE PER PRODUZIONE (SavedModel) ---
# Se hai bisogno di una cartella (formato SavedModel) per TensorFlow Serving o TFLite, 
# Keras 3 richiede il metodo .export() invece di .save().
# Questo risolve il ValueError che hai ricevuto precedentemente.
model.export('my_saved_model_folder') 

# --- 3. SALVATAGGIO DEI SOLI PESI (.weights.h5) ---
# Utile se vuoi salvare solo i parametri numerici (es. durante i checkpoint).
# L'estensione .weights.h5 è lo standard attuale per evitare confusione.
model.save_weights('model_params.weights.h5')

print("Sistemi di persistenza completati con successo.")

# --- CARICAMENTO PER INFERENZA ---

# Caso A: Caricamento modello completo dal file nativo
# Non è necessario ridefinire il modello nel codice Python.
new_model = keras.models.load_model('my_model.keras')

# Caso B: Caricamento dei soli pesi in un'architettura esistente
# Richiede che il modello sia stato prima creato identico all'originale.
weights_model = create_model()
weights_model.load_weights('model_params.weights.h5')

# --- PREDIZIONE ---
# Generiamo un dato casuale (batch_size=1, features=10)
data_point = np.random.random((1, 10)).astype("float32")

# Esecuzione della predizione
# verbose=0 evita di stampare la barra di caricamento per una singola operazione
prediction = new_model.predict(data_point, verbose=0)

print(f"Risultato della predizione: {prediction[0][0]:.4f}")