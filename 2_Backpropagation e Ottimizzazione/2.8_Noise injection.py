import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 1. PREPARAZIONE DEL DATASET ORIGINALE
# Generiamo un dataset non lineare (due mezzelune)
X_orig, y_orig = make_moons(n_samples=200, noise=0.05, random_state=42)

def inject_gaussian_noise(data, sigma=0.1):
    """
    Applica rumore gaussiano additivo al dataset.
    Parametri:
    - data: il tensore o array originale
    - sigma: deviazione standard del rumore (intensità)
    """
    # Generiamo il rumore con la stessa forma del dataset originale
    # Media = 0, Deviazione Standard = sigma
    noise = np.random.normal(loc=0.0, scale=sigma, size=data.shape)
    
    # Restituiamo il dato "aumentato"
    return data + noise

# 2. GENERAZIONE DATI AUMENTATI
# Creiamo tre versioni con intensità di rumore crescente
X_noise_low = inject_gaussian_noise(X_orig, sigma=0.05)
X_noise_med = inject_gaussian_noise(X_orig, sigma=0.15)
X_noise_high = inject_gaussian_noise(X_orig, sigma=0.30)

# 3. VISUALIZZAZIONE COMPARATIVA
plt.figure(figsize=(15, 5))

titles = ['Originale', 'Rumore Lieve (0.05)', 'Rumore Medio (0.15)', 'Rumore Alto (0.30)']
datasets = [X_orig, X_noise_low, X_noise_med, X_noise_high]

for i, ds in enumerate(datasets):
    plt.subplot(1, 4, i+1)
    plt.scatter(ds[:, 0], ds[:, 1], c=y_orig, cmap='viridis', edgecolors='k', alpha=0.7)
    plt.title(titles[i])
    plt.axis('equal')

plt.tight_layout()
plt.show()

print("Analisi: Notate come all'aumentare di sigma, i punti 'esplorano' lo spazio circostante.")