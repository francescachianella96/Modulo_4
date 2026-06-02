import tensorflow as tf
import keras
import numpy as np
import shutil

# 1. DEFINIZIONE ARCHITETTURA E TRAINING
def create_model():
    model = keras.Sequential([
        keras.Input(shape=(10,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# Prepariamo i dati e addestriamo il modello originale
X_train = np.random.random((1000, 10)).astype("float32")
y_train = np.random.randint(2, size=(1000, 1)).astype("float32")

model_orig = create_model()
model_orig.fit(X_train, y_train, epochs=1, verbose=0)

# Dati per il test di integrità (Inferenza)
X_test = np.random.random((5, 10)).astype("float32")
target_pred = model_orig.predict(X_test, verbose=0)

print("--- FASE 1: SALVATAGGIO NEI 3 FORMATI ---")

# METODO 1: Modello Completo (.keras) - IL NUOVO STANDARD
# Salva tutto: architettura, pesi e configurazione ottimizzatore.
model_orig.save("modello_completo.keras")
print("1. Salvato: modello_completo.keras")

# METODO 2: Soli Pesi (.weights.h5) - PER CHECKPOINT
# Salva solo i numeri. È il file più leggero, ma richiede il codice dell'architettura per essere ricaricato.
model_orig.save_weights("soli_pesi.weights.h5")
print("2. Salvato: soli_pesi.weights.h5")

# METODO 3: Esportazione (SavedModel Folder) - PER DEPLOYMENT
# Crea una cartella ottimizzata per TensorFlow Serving o TFLite.
model_orig.export("cartella_export")
print("3. Esportato: cartella_export/\n")



print("--- FASE 2: CARICAMENTO E INFERENZA (CORRETTA) ---")

# CARICAMENTO 1: Dal file .keras (Standard)
model_1 = keras.models.load_model("modello_completo.keras")
pred_1 = model_1.predict(X_test, verbose=0)

# CARICAMENTO 2: Dai soli pesi (Richiede architettura)
model_2 = create_model() 
model_2.load_weights("soli_pesi.weights.h5")
pred_2 = model_2.predict(X_test, verbose=0)

# CARICAMENTO 3: Dalla cartella di esportazione (SavedModel)
# tf.saved_model.load restituisce un oggetto con le "signatures" di serving
model_3 = tf.saved_model.load("cartella_export")

# Per eseguire l'inferenza dobbiamo accedere alla firma predefinita
infer = model_3.signatures["serving_default"]

# Nota: SavedModel si aspetta un tensore e restituisce un dizionario
# 'keras_tensor' è il nome di default dell'input definito da Keras nell'export
pred_3_dict = infer(tf.constant(X_test))
# Estraiamo il risultato (il nome della chiave di solito è 'output_0' o simile)
pred_3 = list(pred_3_dict.values())[0] 

print("\n--- FASE 3: VERIFICA RISULTATI ---")

def check(name, pred):
    diff = np.max(np.abs(target_pred - pred))
    status = "OK" if diff < 1e-5 else "ERRORE"
    print(f"{name}: Differenza max {diff:.8f} -> {status}")

check("Metodo .keras      ", pred_1)
check("Metodo .weights.h5 ", pred_2)
check("Metodo .export     ", pred_3.numpy())