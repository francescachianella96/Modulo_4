import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Calcola l'errore quadratico medio (MSE) per la regressione.
    """
    return np.mean(np.square(y_true - y_pred))

def cross_entropy_loss(y_true, y_pred):
    """
    Calcola la Cross-Entropy per la classificazione.
    y_true: vettore one-hot (es: [1, 0, 0])
    y_pred: probabilità della softmax (es: [0.9, 0.05, 0.05])
    """
    # Epsilon serve a evitare log(0) che restituirebbe -inf
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Formula: -somma(y_reale * log(y_predetto))
    loss = -np.sum(y_true * np.log(y_pred))
    return loss

# --- ESEMPIO DI UTILIZZO ---

# Caso Regressione (Prezzo case in migliaia di euro)
prezzo_reale = np.array([200, 150, 300])
prezzo_predetto = np.array([210, 140, 250]) # Il modello sbaglia di molto sull'ultima casa

mse_val = mean_squared_error(prezzo_reale, prezzo_predetto)
print(f"Errore MSE: {mse_val:.2f}")

# Caso Classificazione (Cane, Gatto, Pesce)
target = np.array([1, 0, 0]) # È un Cane
predizione = np.array([0.9, 0.05, 0.05]) # Il modello è sicuro al 90%

ce_val = cross_entropy_loss(target, predizione)
print(f"Errore Cross-Entropy: {ce_val:.4f}")