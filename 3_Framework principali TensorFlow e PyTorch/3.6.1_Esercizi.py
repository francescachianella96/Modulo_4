import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# =================================================================
# 1. PREPARAZIONE DEL DATASET
# =================================================================
X, y = make_classification(n_samples=2000, n_features=30, n_informative=20, 
                           n_redundant=10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# =================================================================
# 2. CUSTOM CALLBACK: LOG LR E MONITORAGGIO PESI
# =================================================================
class EnhancedLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # 1. Recupero Learning Rate
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        # 2. Monitoraggio Pesi (Prendiamo il primo strato Dense dopo l'Input)
        # Calcoliamo la media dei valori assoluti dei pesi per vedere quanto sono "grandi"
        weights, biases = self.model.layers[0].get_weights()
        avg_weight = np.mean(np.abs(weights))
        
        print(f"\n - [INFO] Fine Epoca {epoch+1}:")
        print(f"   > Learning Rate: {lr:.6f}")
        print(f"   > Media abs pesi (Layer 1): {avg_weight:.6f}")

# =================================================================
# 3. CONFIGURAZIONE CALLBACK
# =================================================================

# Salva il miglior modello
checkpoint_cb = ModelCheckpoint(
    filepath='miglior_modello.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Riduce il LR se la loss stalla per 3 epoche
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# STOPPA il modello se non migliora per 10 epoche
early_stopping_cb = EarlyStopping(
    monitor='val_loss',
    patience=10,        # Numero di epoche da aspettare
    mode='min',
    restore_best_weights=True, # Al termine, ripristina i pesi migliori invece degli ultimi
    verbose=1
)

# =================================================================
# 4. COSTRUZIONE E TRAINING
# =================================================================
model = Sequential([
    Input(shape=(30,)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\nInizio addestramento con Early Stopping e Monitoraggio Pesi...")

history = model.fit(
    X_train, y_train,
    epochs=100, # Aumentiamo le epoche potenziali, tanto lo stop è automatico
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stopping_cb, EnhancedLogger()],
    verbose=0 
)