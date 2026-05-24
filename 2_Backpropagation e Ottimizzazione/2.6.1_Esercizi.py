import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def train_on_dataset(dataset, batch_size, epochs=20, lr=0.01):
    """Funzione di training flessibile per batch size e dataset specifici."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    loss_history = []
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
    return loss_history

# 1. GENERAZIONE DEI TRE DATASET
sizes = [32, 320, 1000]
datasets = {}

for n in sizes:
    X = torch.randn(n, 1)
    y = 2 * X + 0.3 * torch.randn(n, 1) # Rumore ridotto per chiarezza
    datasets[n] = TensorDataset(X, y)

# 2. ESECUZIONE DEL CONFRONTO
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, n in enumerate(sizes):
    ds = datasets[n]
    
    # Eseguiamo i tre scenari
    hist_online = train_on_dataset(ds, batch_size=1)
    hist_mini   = train_on_dataset(ds, batch_size=32)
    hist_full   = train_on_dataset(ds, batch_size=n) # Full Batch = dimensione dataset
    
    # Plotting
    axes[i].plot(hist_online, label='Stochastic (B=1)', alpha=0.4, color='red')
    axes[i].plot(hist_mini, label='Mini-batch (B=32)', alpha=0.8, color='green')
    axes[i].plot(hist_full, label='Full Batch (B=N)', linewidth=2, color='blue')
    
    axes[i].set_title(f"Dataset Size: {n}")
    axes[i].set_xlabel("Iterazioni (Weight Updates)")
    axes[i].set_yscale('log')
    if i == 0: axes[i].set_ylabel("MSE Loss")
    axes[i].legend()
    axes[i].grid(True, alpha=0.2)

plt.tight_layout()
plt.show()