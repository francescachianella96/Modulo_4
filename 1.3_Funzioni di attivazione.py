import numpy as np

def sigmoid(z):
    """Schiaccia l'input nell'intervallo (0, 1)"""
    return 1 / (1 + np.exp(-z))

def tanh(z):
    """Schiaccia l'input nell'intervallo (-1, 1)"""
    return np.tanh(z)

def relu(z):
    """Soglia a zero: restituisce z se positivo, altrimenti 0"""
    return np.maximum(0, z)

# --- TEST DELLE FUNZIONI ---

z_test = 2.0  # Valore di esempio (somma pesata)

print(f"Risultati per z = {z_test}:")
print(f"1. Sigmoide: {sigmoid(z_test):.4f}")
print(f"2. Tanh:     {tanh(z_test):.4f}")
print(f"3. ReLU:     {relu(z_test):.4f}")

# Esempio con valore negativo
z_neg = -1.5
print(f"\nRisultati per z = {z_neg}:")
print(f"1. Sigmoide: {sigmoid(z_neg):.4f}")
print(f"2. Tanh:     {tanh(z_neg):.4f}")
print(f"3. ReLU:     {relu(z_neg):.4f}")