import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# --- 1. PREPARAZIONE DATI ---
# Generazione di 200 punti casuali per X
X = torch.randn(200, 1) * 5 
# Funzione target: y = 2x (senza rumore)
y_target = 2 * X 
# Creazione del DataLoader per gestire i batch durante l'addestramento
dataset = TensorDataset(X, y_target)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- 2. MODELLO MULTI-STRATO ---
class DeepInspectorNet(nn.Module):
    def __init__(self):
        super(DeepInspectorNet, self).__init__()
        # Definizione dei layer lineari
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)
        self.relu = nn.ReLU() # Funzione di attivazione per introdurre non-linearità

    def forward(self, x):
        # Passaggio dei dati attraverso la rete
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

model = DeepInspectorNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss() # Errore Quadratico Medio (MSE)

# --- 3. CICLO DI TRAINING ---
epochs = 50
print(f"{'Epoca':<10} | {'Loss Media':<10}")
print("-" * 25)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in loader:
        optimizer.zero_grad()          # Azzera i gradienti accumulati
        preds = model(batch_X)         # Calcola le predizioni
        loss = criterion(preds, batch_y) # Calcola l'errore
        loss.backward()                # Calcola i gradienti (Backpropagation)
        optimizer.step()               # Aggiorna i pesi del modello
        total_loss += loss.item()

    # Stampa l'andamento ogni 10 epoche
    if (epoch + 1) % 5 == 0:
        avg_loss = total_loss / len(loader)
        print(f"{epoch+1:<5} | {avg_loss:<5.4f}")

# --- 4. GRAFICO FINALE ---
plt.figure(figsize=(10, 6))

# Modalità valutazione per generare le predizioni finali
model.eval()
with torch.no_grad():
    final_preds = model(X)

# Visualizzazione dei dati reali rispetto alla curva appresa
plt.scatter(X.numpy(), y_target.numpy(), color='gray', alpha=0.5, label='Dati con Rumore')
plt.plot(X.numpy(), final_preds.numpy(), color='green', linewidth=3, label='Predizione Modello')
plt.title('Risultato della Regressione (Approssimazione della retta)')
plt.xlabel('Input (X)')
plt.ylabel('Output (Y)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nAddestramento terminato e grafico visualizzato.")