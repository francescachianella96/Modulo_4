import tensorflow as tf
import numpy as np

# --- CONFIGURAZIONE DATASET ---
X_raw = np.random.uniform(0, 255, (10000, 20)).astype(np.float32)
y_raw = np.random.randint(0, 2, (10000, 1)).astype(np.float32)
dataset = tf.data.Dataset.from_tensor_slices((X_raw, y_raw))

# --- LOGICA DI PIPELINE OTTIMIZZATA ---

# 1. MAP: Operazione costosa (CPU)
# Qui facciamo calcoli matematici e generiamo numeri casuali (rumore).
dataset = dataset.map(lambda x, y: (x / 255.0 + tf.random.normal(tf.shape(x), stddev=0.01), y), 
                      num_parallel_calls=tf.data.AUTOTUNE)

# 2. CACHE: Il "Salva-Risultati"
# POSIZIONE: Dopo il MAP.
# PERCHÉ: Vogliamo memorizzare il dato già pulito e normalizzato. 
# In questo modo, dalla seconda epoca in poi, il calcolo del 'map' sopra 
# viene saltato completamente, risparmiando cicli di CPU preziosi.
dataset = dataset.cache()



# 3. SHUFFLE: Il "Mescolatore"
# POSIZIONE: Dopo il CACHE.
# PERCHÉ: Se lo mettessimo PRIMA del cache, l'ordine casuale verrebbe memorizzato (congelato).
# Invece, mettendolo DOPO, il cache fornisce i dati (veloce), e lo shuffle li rimescola 
# in modo diverso a ogni epoca. Questo garantisce la varietà statistica necessaria 
# per evitare che il modello impari la sequenza dei dati invece dei pattern.
dataset = dataset.shuffle(buffer_size=1000)



# 4. BATCH e PREFETCH: Gli "Organizzatori"
# Una volta che i dati sono estratti e mescolati, li raggruppiamo e 
# prepariamo il batch successivo in anticipo (Prefetch).
dataset = dataset.batch(32).prefetch(buffer_size=tf.data.AUTOTUNE)

# --- ESECUZIONE ---
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(dataset, epochs=3)