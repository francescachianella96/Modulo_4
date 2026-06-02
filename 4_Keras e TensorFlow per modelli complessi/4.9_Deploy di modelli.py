import tensorflow as tf
import numpy as np
import os

# --- 1. DEFINIZIONE DEL MODELLO (Keras 3 & Functional/Sequential API) ---
# Nel 2025, definire esplicitamente l'Input layer è fondamentale per Keras 3.
# Questo garantisce che il modello sia "tracciabile" per diversi backend (JAX, PyTorch, TF)
# e facilita la conversione in grafi statici come TFLite.
model = tf.keras.Sequential([
    # Definiamo la forma dell'input: un vettore di 10 elementi.
    tf.keras.layers.Input(shape=(10,), name="input_layer"),
    
    # Layer denso con 64 neuroni e attivazione ReLU per apprendere pattern non lineari.
    tf.keras.layers.Dense(64, activation='relu', name="hidden_layer"),
    
    # Layer di output con 10 neuroni (classi) e attivazione Softmax per le probabilità.
    tf.keras.layers.Dense(10, activation='softmax', name="output_layer")
])

# Compilazione: usiamo Adam come ottimizzatore standard e Sparse Categorical Crossentropy
# poiché ci aspettiamo etichette intere (0, 1, 2...) invece di vettori one-hot.
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# --- 2. PROCESSO DI CONVERSIONE IN TENSORFLOW LITE ---
# Carichiamo il modello Keras nel convertitore TFLite.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Applichiamo la "Dynamic Range Quantization". 
# Questa tecnica riduce il peso dei pesi del modello da float32 a int8 durante il salvataggio,
# diminuendo le dimensioni del file di ~4 volte e velocizzando l'esecuzione senza 
# perdere eccessiva precisione.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Generazione del file .tflite (il cuore del modello per dispositivi Edge/Mobile)
tflite_model = converter.convert()

# Salvataggio fisico su disco.
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Modello convertito con successo! Dimensione: {len(tflite_model) / 1024:.2f} KB")

# --- 3. INFERENZA MODERNA (Signature Runner) ---
# Invece di manipolare manualmente gli indici dei tensori (metodo legacy), 
# usiamo le 'Signatures', che permettono di chiamare il modello come una funzione Python.

# Inizializziamo l'interprete caricando il file dal disco.
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")

# Il Signature Runner mappa automaticamente gli input e gli output tramite i nomi dei layer.
# È il metodo più sicuro e leggibile disponibile nel 2025.
prediction_fn = interpreter.get_signature_runner()

# Creiamo un dato di test casuale. Nota: la forma deve essere (batch_size, input_dim), quindi (1, 10).
test_input = np.random.random_sample((1, 10)).astype(np.float32)

# Esecuzione del modello:
# Passiamo l'input usando il nome del layer (spesso 'input_1' o quello definito sopra).
# Il risultato è un dizionario contenente i tensori di output.
output_data = prediction_fn(input_layer=test_input) 

# Estraiamo il risultato dal dizionario usando la chiave corrispondente all'output.
output_key = list(output_data.keys())[0]
probabilities = output_data[output_key]

# Individuiamo la classe con il valore di probabilità più alto.
predicted_class = np.argmax(probabilities)

print(f"Risultato dell'inferenza: Classe {predicted_class}")
print(f"Vettore probabilità: {probabilities}")