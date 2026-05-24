import numpy as np

# 1. Funzioni di supporto
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # La derivata della sigmoide espressa in funzione dell'output
    return x * (1 - x)

# 2. Inizializzazione Dati (XOR semplificato)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inizializzazione Pesi e Bias (Casuale)
np.random.seed(42)
weights_hidden = np.random.rand(2, 3)
bias_hidden = np.random.randn(1, 3)
weights_output = np.random.rand(3, 1)
bias_output = np.random.randn(1, 1)

learning_rate = 0.5

# 3. Training Loop (Singola Epoca per analisi)
for epoch in range(3):
    # --- FORWARD PASS ---
    # Layer nascosto
    hidden_input = np.dot(X, weights_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    
    # Layer di output
    final_input = np.dot(hidden_output, weights_output) + bias_output
    final_output = sigmoid(final_input)
    
    # Calcolo Errore (MSE)
    error = y - final_output
    
    # --- BACKWARD PASS (Backpropagation) ---
    
    # PUNTO CHIAVE 1: Gradiente sull'ultimo layer (Delta Output)
    # Calcolo: (Target - Predizione) * Derivata_Sigmoide
    d_output = error * sigmoid_derivative(final_output)
    
    # PUNTO CHIAVE 2: Propagazione al layer nascosto (Delta Hidden)
    # L'errore viene "pesato" dai pesi del layer di output
    error_hidden = d_output.dot(weights_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    # PUNTO CHIAVE 3: Aggiornamento Parametri
    # I pesi vengono modificati: peso = peso + (input * delta * learning_rate)
    weights_output += hidden_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
    weights_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    print("Matrice Pesi Hidden (2x3):")
    print(weights_hidden)

print("Aggiornamento completato con successo.")