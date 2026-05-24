def gradient_descent_update(w, grad, eta):
    """
    Esegue un singolo step di aggiornamento del peso.
    w: valore attuale del peso
    grad: valore del gradiente (derivata della loss rispetto a w)
    eta: learning rate
    """
    # La formula fondamentale: w = w - eta * grad
    w_new = w - eta * grad
    
    return w_new

# --- SIMULAZIONE ---

peso_iniziale = 0.5
gradiente_calcolato = 2.4  # Supponiamo che la pendenza sia verso l'alto
learning_rate = 0.1

# Primo aggiornamento
peso_aggiornato = gradient_descent_update(peso_iniziale, gradiente_calcolato, learning_rate)

print(f"Peso prima dell'aggiornamento: {peso_iniziale}")
print(f"Valore del gradiente: {gradiente_calcolato}")
print(f"Peso dopo l'aggiornamento: {peso_aggiornato:.2f}")

# Nota: Il peso è diminuito perché il gradiente era positivo.