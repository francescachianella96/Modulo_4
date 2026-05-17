import numpy as np

def min_max_scaler(data):
    """
    Trasforma i dati nell'intervallo [0, 1].
    Formula: (x - min) / (max - min)
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    # Calcolo normalizzato
    scaled_data = (data - data_min) / (data_max - data_min)
    return scaled_data

def standardizer(data):
    """
    Trasforma i dati con media 0 e deviazione standard 1.
    Formula: (x - mean) / std
    """
    mean = np.mean(data)
    std = np.std(data)
    
    # Calcolo standardizzato (Z-score)
    standardized_data = (data - mean) / std
    return standardized_data

# --- TEST DELLE FUNZIONI ---

# Immaginiamo i prezzi di 5 prodotti in euro
prezzi_grezzi = np.array([50, 150, 200, 450, 900])

prezzi_normalizzati = min_max_scaler(prezzi_grezzi)
prezzi_standardizzati = standardizer(prezzi_grezzi)

print(f"Dati Originali: {prezzi_grezzi}")
print(f"Min-Max [0, 1]: {np.round(prezzi_normalizzati, 3)}")
print(f"Standardizzati: {np.round(prezzi_standardizzati, 3)}")

# Nota: Osservate come nella Standardizzazione i valori possano essere negativi
