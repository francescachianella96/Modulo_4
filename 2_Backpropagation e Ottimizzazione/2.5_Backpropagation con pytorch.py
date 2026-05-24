import torch

# 1. PREPARAZIONE DEI DATI
# Creiamo input (x) e target (y) come tensori costanti
x = torch.tensor([[1.5], [2.0], [3.0]], dtype=torch.float32)
y_target = torch.tensor([[0.5], [0.8], [1.2]], dtype=torch.float32)

# 2. DEFINIZIONE DEI PARAMETRI (requires_grad=True)
# Inizializziamo il peso W e il bias b casualmente. 
# Fondamentale attivare il tracciamento dei gradienti.
W = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"Inizializzazione - W: {W.item():.4f}, b: {b.item():.4f}\n")

# 3. FORWARD PASS
# Costruiamo il grafo computazionale dinamico
y_pred = x @ W + b

# 4. CALCOLO DELLA LOSS
# Usiamo l'errore quadratico medio (MSE)
loss = torch.mean((y_pred - y_target)**2)
print(f"Loss calcolata: {loss.item():.6f}")

# 5. BACKWARD PASS (Motore Autograd)
# Calcola il gradiente di 'loss' rispetto a tutti i parametri con requires_grad=True
loss.backward()

# 6. ISPEZIONE DEI GRADIENTI
# I gradienti vengono accumulati negli attributi .grad
print("\n--- Ispezione dei Gradienti ---")
print(f"Gradiente rispetto a W (dL/dW): {W.grad.item():.6f}")
print(f"Gradiente rispetto a b (dL/db): {b.grad.item():.6f}")

# 7. AGGIORNAMENTO MANUALE (Teoria del Gradient Descent)
# Usiamo torch.no_grad() per evitare che l'operazione di aggiornamento 
# venga tracciata nel grafo (causerebbe errori o loop infiniti)
lr = 0.01
with torch.no_grad():
    W -= lr * W.grad
    b -= lr * b.grad

print(f"\nParametri dopo un passo di ottimizzazione:")
print(f"W: {W.item():.4f}, b: {b.item():.4f}")