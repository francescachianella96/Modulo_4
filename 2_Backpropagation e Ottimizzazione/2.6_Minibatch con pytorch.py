import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 1. GENERAZIONE DATASET SINTETICO
# Creiamo 1000 punti: y = 2x + rumore
X = torch.randn(1000, 1)
y = 2 * X + 0.5 * torch.randn(1000, 1)
dataset = TensorDataset(X, y)

def train_with_batch_size(batch_size, epochs=5):
    """Funzione per addestrare il modello con una specifica dimensione di batch."""
    # Definiamo il DataLoader con la dimensione desiderata
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modello lineare semplice
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    loss_history = []
    
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad() # Reset dei gradienti
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, targets) # Calcolo errore
            loss.backward() # Backpropagation
            optimizer.step() # Aggiornamento pesi
            
            # Salviamo la loss per ogni aggiornamento (iterazione)
            loss_history.append(loss.item())
            
    return loss_history

# 2. ESECUZIONE DEI TRE SCENARI
print("Addestramento in corso...")
# Online: batch_size=1
history_online = train_with_batch_size(batch_size=1)
# Mini-batch: batch_size=32
history_mini = train_with_batch_size(batch_size=32)
# Full Batch: batch_size=1000
history_batch = train_with_batch_size(batch_size=1000)

# 3. VISUALIZZAZIONE DEI RISULTATI
plt.figure(figsize=(12, 6))
plt.plot(history_online, label='Online (Batch=1)', alpha=0.4, color='red')
plt.plot(history_mini, label='Mini-Batch (Batch=32)', alpha=0.8, color='green')
plt.plot(history_batch, label='Full Batch (Batch=1000)', linewidth=2, color='blue')

plt.title("Confronto Stabilità Loss per Dimensione Batch")
plt.xlabel("Iterazioni (Aggiornamenti dei pesi)")
plt.ylabel("MSE Loss")
plt.yscale('log') # Scala logaritmica per vedere meglio le differenze
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()

print("Analisi completata. Osserva come la curva blu è liscia, mentre quella rossa è rumorosa.")