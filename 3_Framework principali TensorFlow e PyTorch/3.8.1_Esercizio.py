import tensorflow as tf
import torch
import torch.nn as nn

# =============================================================================
# LOGICA DI CALCOLO MATEMATICO DEI PARAMETRI (PASSO DOPO PASSO):
# 
# 1. LAYER NASCOSTO (Hidden Layer):
#    - Input: 10 feature
#    - Neuroni: 32
#    - Pesi (Weights): 10 input * 32 neuroni = 320 pesi
#    - Bias: 1 per ogni neurone = 32 bias
#    - Totale Layer 1: 320 + 32 = 352 parametri
#
# 2. LAYER DI OUTPUT:
#    - Input: 32 (provenienti dal layer precedente)
#    - Classi (Neuroni): 2
#    - Pesi (Weights): 32 input * 2 neuroni = 64 pesi
#    - Bias: 1 per ogni classe = 2 bias
#    - Totale Layer 2: 64 + 2 = 66 parametri
#
# 3. CONTEGGIO TOTALE:
#    - 352 (Layer 1) + 66 (Layer 2) = 418 parametri totali
# =============================================================================

# --- IMPLEMENTAZIONE KERAS ---

# Definiamo l'architettura Sequential
# Il bias è incluso di default in tf.keras.layers.Dense
keras_model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(10,), activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Estraiamo il numero di parametri totali tramite il metodo integrato
keras_total_params = keras_model.count_params()


# --- IMPLEMENTAZIONE PYTORCH ---

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # nn.Linear(in_features, out_features) include il bias di default
        self.layer1 = nn.Linear(10, 32)
        self.layer2 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        return self.layer2(x)

pt_model = MyModel()

# Calcoliamo i parametri iterando su tutti i tensori del modello
# p.numel() restituisce il numero totale di elementi in ogni tensore (Weight e Bias)
pt_total_params = sum(p.numel() for p in pt_model.parameters())


# --- VERIFICA E STAMPA ---

print(f"Conteggio Parametri Keras:   {keras_total_params}")
print(f"Conteggio Parametri PyTorch: {pt_total_params}")

if keras_total_params == pt_total_params == 418:
    print("\nSUCCESSO: Entrambi i modelli hanno 418 parametri, come previsto dal calcolo.")
else:
    print("\nERRORE: I parametri non corrispondono al valore atteso.")