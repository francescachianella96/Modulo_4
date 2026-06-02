import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. PREPARAZIONE DEI DATI
X, y = make_classification(n_samples=1000, n_features=30, n_informative=15, 
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. DEFINIZIONE ARCHITETTURA
model = Sequential([
    Input(shape=(30,)), 
    Dense(32, activation='relu', name="Hidden_Layer_1"),
    Dropout(0.2),
    Dense(16, activation='relu', name="Hidden_Layer_2"),    # ho provato ad aggiungere un dropout anche sul secondo neurone, ma l'accuratezza si ferma, anche se la loss sul validation migliora. La scelta migliore è rimanere a 16.
    Dense(1, activation='sigmoid', name="Output_Layer")
])

# 3. COMPILAZIONE
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. ISPEZIONE
model.summary()

# 5. ADDESTRAMENTO
print("\nInizio addestramento...")
# Salviamo l'output di fit nella variabile 'history' per poter tracciare i grafici
history = model.fit(X_train, y_train, 
                    epochs=50,             # Provato sia con 50 che con 80 epoche. Con 50 epoche viene un'accuracy di 0.90, con 80 epoche di 0.89
                    batch_size=32,         
                    validation_split=0.2,  # Usiamo il 20% del train per la validazione live
                    verbose=1)

# 6. VALUTAZIONE FINALE
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAccuratezza finale sul Test Set: {acc:.4f}")

# 7. VISUALIZZAZIONE DELLE CURVE DI APPRENDIMENTO
# Creiamo una figura con due grafici (uno per la loss e uno per l'accuratezza)
plt.figure(figsize=(14, 5))

# --- GRAFICO ACCURACY ---
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', linewidth=2)
plt.title('Andamento Accuratezza (Accuracy)')
plt.xlabel('Epoche')
plt.ylabel('Accuratezza')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

# --- GRAFICO LOSS ---
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='#d62728', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='#9467bd', linewidth=2)
plt.title('Andamento Errore (Loss)')
plt.xlabel('Epoche')
plt.ylabel('Perdita')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()