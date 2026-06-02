import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from tensorflow.keras.utils import plot_model

# =============================================================================
# 1. PREPARAZIONE DEI DATI (Dataset Sintetico)
# =============================================================================

# Creiamo 500 esempi per ogni sorgente dati.
# input_tecnico_data: 10 colonne (es. parametri audio, frequenze).
input_tecnico_data = np.random.random((500, 10))

# input_social_data: 5 colonne (es. numero di condivisioni, commenti).
input_social_data = np.random.random((500, 5))

# targets: L'obiettivo è binario (0 o 1), ad esempio "Successo" o "Insuccesso".
# Usiamo shape (500, 1) per corrispondere all'output del neurone finale.
targets = np.random.randint(0, 2, size=(500, 1))

# =============================================================================
# 2. DEFINIZIONE DELL'ARCHITETTURA (API FUNZIONALE)
# =============================================================================

# Utilizziamo layers.Input per dichiarare la forma (shape) dei dati in ingresso.
# name aiuta a identificare i layer nel summary e nei grafici.
input_tecnico = layers.Input(shape=(10,), name="Ingresso_Tecnico")
input_social = layers.Input(shape=(5,), name="Ingresso_Social")

# --- RAMO TECNICO ---
# Elaboriamo i dati tecnici con layer densi. 
# Sintassi funzionale: Layer()(Input) crea un legame diretto tra i nodi.
x = layers.Dense(16, activation='relu')(input_tecnico)
x = layers.Dense(8, activation='relu')(x) # Riduce a 8 feature estratte

# --- RAMO SOCIAL ---
# Elaboriamo i dati social separatamente. Usiamo meno neuroni perché l'input è più piccolo.
y = layers.Dense(4, activation='relu')(input_social) # Riduce a 4 feature estratte

# --- FUSIONE DEI RAMI (CONCATENATE) ---
# Uniamo i due flussi. Il vettore risultante avrà dimensione 8 (da x) + 4 (da y) = 12.
# Questa fase permette alla rete di trovare correlazioni tra dati tecnici e social.
combined = layers.concatenate([x, y])

# --- TESTA DEL MODELLO (HEAD) ---
# Aggiungiamo layer comuni che lavorano sulle informazioni già fuse e filtrate.
z = layers.Dense(8, activation='relu')(combined)

# Output finale: 1 solo neurone con attivazione Sigmoid per dare una probabilità [0, 1].
output_finale = layers.Dense(1, activation='sigmoid', name="Output_Successo")(z)

# =============================================================================
# 3. CREAZIONE E COMPILAZIONE DEL MODELLO
# =============================================================================

# L'oggetto Model definisce i confini del grafo: dove inizia (lista input) e dove finisce.
model = Model(inputs=[input_tecnico, input_social], outputs=output_finale)

# Compiliamo specificando l'ottimizzatore e la funzione di perdita per classificazione binaria.
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Visualizza la struttura: noterai come i due input iniziali convergono verso il layer 'concatenate'.
model.summary()

# =============================================================================
# 4. ADDESTRAMENTO
# =============================================================================

# Quando il modello ha più input, il parametro 'x' deve ricevere una LISTA di array.
# L'ordine nella lista deve corrispondere esattamente all'ordine definito in Model(inputs=[...]).
model.fit(
    x=[input_tecnico_data, input_social_data], 
    y=targets, 
    epochs=5, 
    batch_size=32
)

print("\nModello addestrato con successo gestendo input eterogenei.")
