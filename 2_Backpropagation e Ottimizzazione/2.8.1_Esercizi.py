import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 1. PREPARAZIONE DEL DATASET
X_orig, y_orig = make_moons(n_samples=200, noise=0.05, random_state=42)

# 2. NUOVA FUNZIONE PER IL RUMORE UNIFORME
def inject_uniform_noise(data, limit=0.2):
    """
    Applica rumore UNIFORME al dataset.
    I punti verranno spostati casualmente tra -limit e +limit.
    """
    # Generiamo rumore dove ogni valore ha la stessa probabilità di uscire
    noise = np.random.uniform(low=-limit, high=limit, size=data.shape)
    
    return data + noise

# 3. GENERAZIONE DATI
# Proviamo con un limite di 0.1 e uno più 'estremo' di 0.3
X_unif_low = inject_uniform_noise(X_orig, limit=0.1)
X_unif_high = inject_uniform_noise(X_orig, limit=0.3)

# 4. VISUALIZZAZIONE
plt.figure(figsize=(12, 5))

titles = ['Originale', 'Rumore Uniforme (0.1)', 'Rumore Uniforme (0.3)']
datasets = [X_orig, X_unif_low, X_unif_high]

for i, ds in enumerate(datasets):
    plt.subplot(1, 3, i+1)
    plt.scatter(ds[:, 0], ds[:, 1], c=y_orig, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.title(titles[i])
    plt.axis('equal')

plt.tight_layout()
plt.show()

print("Confronto fatto! Noti come i punti sembrano quasi formare dei 'quadratini' attorno alla posizione originale?")