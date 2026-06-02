import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# =================================================================
# 1. PREPARAZIONE DEL DATASET
# =================================================================
# Creiamo un problema di classificazione binaria complesso per simulare 
# un plateau nella loss dove le callback dovranno intervenire.
X, y = make_classification(n_samples=2000, n_features=30, n_informative=20, 
                           n_redundant=10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# =================================================================
# 2. CREAZIONE DI UNA CUSTOM CALLBACK (CORRETTA)
# =================================================================
class SimpleLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Accediamo al learning rate usando il nome attributo corretto
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"\n - [INFO] Fine Epoca {epoch+1}: il Learning Rate attuale è {lr:.6f}")

# =================================================================
# 3. CONFIGURAZIONE DELLE CALLBACK STANDARD
# =================================================================

# ModelCheckpoint: salva il modello su disco solo se la val_loss migliora.
# Questo garantisce che a fine training avremo il file con i pesi ottimali.
checkpoint_cb = ModelCheckpoint(
    filepath='miglior_modello.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# ReduceLROnPlateau: riduce il LR se la val_loss non scende per 3 epoche.
# È come scalare la marcia quando l'auto fatica in salita per avere più coppia.
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,       # Riduce il LR dell'80% (moltiplica per 0.2)
    patience=3,       # Aspetta 3 epoche di stallo
    min_lr=1e-6,      # Non scendere sotto questo valore
    verbose=1
)

# =================================================================
# 4. COSTRUZIONE E TRAINING DEL MODELLO
# =================================================================
model = Sequential([
    Input(shape=(30,)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Passiamo la lista delle callback al metodo fit
print("\nInizio addestramento con automazione attiva...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_cb, reduce_lr_cb, SimpleLogger()],
    verbose=0 # Usiamo la nostra callback e il verbose del checkpoint per i log
    ) 