import tensorflow as tf
import numpy as np
import os

# Configurazione per pulizia output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size / 1024  # Ritorna la dimensione in KB

# --- 1. CREAZIONE E ADDESTRAMENTO (Dati reali sintetici) ---
print("--- 1. Fase di Addestramento ---")
# Creiamo un dataset sintetico: 1000 campioni, 10 feature, 3 classi
X_train = np.random.rand(1000, 10).astype(np.float32)
y_train = np.random.randint(0, 3, size=(1000,))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,), name="input_layer"),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax', name="output_layer")
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Addestramento rapido
model.fit(X_train, y_train, epochs=5, verbose=0)
print("Modello addestrato con successo.")

# Salvataggio modello originale (Keras standard .keras)
model.save("original_model.keras")

# --- 2. CONVERSIONE E QUANTIZZAZIONE ---
print("\n--- 2. Conversione e Quantizzazione ---")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Applichiamo la Dynamic Range Quantization (Standard 2025)
# Questo converte i pesi da float32 (4 byte) a int8 (1 byte)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Salvataggio del modello ottimizzato
tflite_filename = "model_quantized.tflite"
with open(tflite_filename, 'wb') as f:
    f.write(tflite_quantized_model)

# --- 3. CONFRONTO PESO IN MEMORIA ---
size_keras = get_file_size("original_model.keras")
size_tflite = get_file_size(tflite_filename)
risparmio = (1 - (size_tflite / size_keras)) * 100

print(f"{'Formato':<20} | {'Dimensione (KB)':<15}")
print("-" * 40)
print(f"{'Keras Originale':<20} | {size_keras:>15.2f}")
print(f"{'TFLite Quantizzato':<20} | {size_tflite:>15.2f}")
print("-" * 40)
print(f"Riduzione del peso: {risparmio:.2f}%")

# --- 4. INFERENZA CON INTERPRETE (Signature Runner) ---
print("\n--- 4. Inferenza on-device ---")
interpreter = tf.lite.Interpreter(model_path=tflite_filename)
# Il signature runner è il metodo consigliato nel 2025 per evitare errori di mapping
prediction_fn = interpreter.get_signature_runner()

# Generiamo un dato di test casuale (1 campione con 10 feature)
sample_input = np.random.rand(1, 10).astype(np.float32)

# Esecuzione
# Passiamo l'input usando il nome definito nell'Input layer ('input_layer')
output = prediction_fn(input_layer=sample_input)

# Estrazione risultato
output_key = list(output.keys())[0]
probabilities = output[output_key]
predicted_class = np.argmax(probabilities)

print(f"Input test: {sample_input}")
print(f"Classe predetta: {predicted_class}")
print(f"Probabilità: {probabilities}")