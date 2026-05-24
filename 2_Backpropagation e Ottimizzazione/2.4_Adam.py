import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam

# 1. Generazione di un dataset sintetico non lineare (due mezzelune)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model():
    """Crea una struttura di rete neurale fissa per il confronto."""
    model = Sequential([
        Dense(16, activation='relu', input_shape=(2,)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# 2. Configurazione degli ottimizzatori
# Usiamo lo stesso Learning Rate iniziale per entrambi per un confronto equo
lr_iniziale = 0.01

# Modello con SGD (Stochastic Gradient Descent classico)
model_sgd = build_model()
model_sgd.compile(optimizer=SGD(learning_rate=lr_iniziale), loss='binary_crossentropy', metrics=['accuracy'])

# Modello con Adam (Ottimizzatore adattivo)
model_adam = build_model()
model_adam.compile(optimizer=Adam(learning_rate=lr_iniziale), loss='binary_crossentropy', metrics=['accuracy'])

# 3. Addestramento
epochs = 50
print("Inizio addestramento SGD...")
history_sgd = model_sgd.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test, y_test))

print("Inizio addestramento Adam...")
history_adam = model_adam.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0, validation_data=(X_test, y_test))

# 4. Visualizzazione dei risultati
plt.figure(figsize=(12, 5))

# Plot della Loss
plt.subplot(1, 2, 1)
plt.plot(history_sgd.history['loss'], label='Loss SGD', color='red', linestyle='--')
plt.plot(history_adam.history['loss'], label='Loss Adam', color='blue')
plt.title('Confronto Loss (Errore)')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend()

# Plot dell'Accuratezza
plt.subplot(1, 2, 2)
plt.plot(history_sgd.history['accuracy'], label='Acc. SGD', color='red', linestyle='--')
plt.plot(history_adam.history['accuracy'], label='Acc. Adam', color='blue')
plt.title('Confronto Accuratezza')
plt.xlabel('Epoca')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()