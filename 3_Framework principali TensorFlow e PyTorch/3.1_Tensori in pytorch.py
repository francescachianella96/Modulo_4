import numpy as np
import torch

# 1. IL PONTE TRA NUMPY E PYTORCH
# Creiamo un array NumPy classico (CPU-only, calcolo scientifico standard)
np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

# Conversione in Tensore: notate come condividono la memoria!
# Se cambiate np_array, cambierà anche tensor_from_np.
tensor_from_np = torch.from_numpy(np_array)

print(f"Tensore creato da NumPy:\n{tensor_from_np}")
print(f"Device attuale: {tensor_from_np.device} (Sempre CPU per NumPy)\n")

# 2. SLICING E INDEXING (NAVIGAZIONE SPAZIALE)
# Creiamo un tensore 3D: immagina 2 matrici, ognuna 3x4
# Shape: [Pagine, Righe, Colonne] -> [2, 3, 4]
t3d = torch.randn(2, 3, 4)

# Estrazione: vogliamo tutte le righe, ma solo la seconda colonna, della prima pagina
# Usiamo l'indice 0 per la prima pagina e 1 per la seconda colonna (zero-based)
slice_1 = t3d[0, :, 1] 

# Uso dell'operatore Ellipsis (...) per dire "prendi tutto il resto"
# Prende l'ultima colonna di tutte le pagine e tutte le righe
slice_2 = t3d[..., -1] 

print(f"Shape originale: {t3d.shape}")
print(f"Shape dello slice (seconda colonna, prima pagina): {slice_1.shape}\n")

# 3. RESHAPE E TRANSPOSE (CAMBIARE LA GEOMETRIA)
# Vogliamo "appiattire" il nostro tensore 3D in una matrice 2D
# Il numero totale di elementi deve restare costante (2*3*4 = 24)
# Usiamo -1 per far calcolare a PyTorch la dimensione mancante
t_flat = t3d.view(2, -1) # Risultato: [2, 12]

# Scambiamo le prime due dimensioni (Transpose)
# Utile se vogliamo passare da [Batch, Canali, Larghezza] a [Canali, Batch, Larghezza]
t_transposed = t3d.transpose(0, 1) # Risultato: [3, 2, 4]

print(f"Shape dopo View (-1): {t_flat.shape}")
print(f"Shape dopo Transpose(0, 1): {t_transposed.shape}")