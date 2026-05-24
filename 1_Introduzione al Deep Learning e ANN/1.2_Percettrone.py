import numpy as np

def heaviside_step(z):
    """
    Implementazione della funzione di attivazione Heaviside.
    Restituisce 1 se l'input è >= 0, altrimenti 0.
    """
    return 1 if z >= 0 else 0

def perceptron_output(inputs, weights, bias):
    """
    Calcola l'output di un singolo neurone (Percettrone).
    Formula: output = step(sum(w_i * x_i) + b)
    """
    # Trasformiamo le liste in array NumPy per usare il prodotto scalare
    x = np.array(inputs)
    w = np.array(weights)
    
    # Calcolo della somma pesata (z = W · x + b)
    # np.dot esegue la sommatoria dei prodotti w_i * x_i
    z = np.dot(w, x) + bias
    
    # Applicazione della funzione di attivazione
    y = heaviside_step(z)
    
    return z, y

# --- ESEMPIO DI UTILIZZO ---

# Definiamo i parametri (es: un neurone che decide se uscire a correre)
# Input: [Meteo_Bello, Ho_Energia]
input_data = [1, 0] 
pesi = [0.8, 0.5]    # Il meteo conta più dell'energia
termine_bias = -0.6  # Soglia di "pigrizia" da superare

z_val, output = perceptron_output(input_data, pesi, termine_bias)

print(f"Somma pesata (z): {z_val:.2f}")
print(f"Decisione finale (Output): {output}")