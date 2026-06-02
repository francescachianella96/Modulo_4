import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from tensorflow.keras.utils import plot_model

# =============================================================================
# 1. PREPARAZIONE DEI DATI (Dataset Sintetico per Regressione)
# =============================================================================

# Creiamo 1000 esempi per tre diverse sorgenti di dati (8 feature ciascuna)
n_samples = 1000
in1_data = np.random.random((n_samples, 8))
in2_data = np.random.random((n_samples, 8))
in3_data = np.random.random((n_samples, 8))

# Target per regressione: un valore continuo (es. prezzo, temperatura)
# Sommiamo casualmente i dati per simulare una relazione
targets = np.sum(in1_data, axis=1) + np.mean(in2_data, axis=1) - np.max(in3_data, axis=1)

# =============================================================================
# 2. DEFINIZIONE DELL'ARCHITETTURA (API FUNZIONALE)
# =============================================================================

# Definizione dei tre ingressi distinti (8 feature l'uno)
input_a = layers.Input(shape=(8,), name="Ingresso_A")
input_b = layers.Input(shape=(8,), name="Ingresso_B")
input_c = layers.Input(shape=(8,), name="Ingresso_C")

# Ogni ingresso passa attraverso il proprio layer Dense da 4 neuroni
# Usiamo 'relu' come attivazione standard per i layer intermedi
branch_a = layers.Dense(4, activation='relu', name="Dense_A")(input_a)
branch_b = layers.Dense(4, activation='relu', name="Dense_B")(input_b)
branch_c = layers.Dense(4, activation='relu', name="Dense_C")(input_c)

# Unione dei tre rami (Concatenazione)
# Il vettore risultante avrà dimensione 4 + 4 + 4 = 12
merged = layers.concatenate([branch_a, branch_b, branch_c], name="Fusione_Rami")

# Output finale per regressione: 1 neurone, attivazione lineare
output_regressione = layers.Dense(1, activation='linear', name="Output_Predizione")(merged)

# Creazione del Modello
model = Model(inputs=[input_a, input_b, input_c], outputs=output_regressione)

# =============================================================================
# 3. COMPILAZIONE E TRAINING
# =============================================================================

# Per la regressione usiamo l'errore quadratico medio (MSE)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Visualizzazione della struttura
model.summary()

# Addestramento: passiamo i tre array nella lista degli input
model.fit(
    x=[in1_data, in2_data, in3_data], 
    y=targets, 
    epochs=10, 
    batch_size=32,
    validation_split=0.2
)

print("\nModello di regressione a 3 ingressi configurato e addestrato.")