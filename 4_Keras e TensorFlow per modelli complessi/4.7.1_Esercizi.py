import tensorflow as tf
import keras
import keras_tuner as kt
import numpy as np

# 1. PREPARAZIONE DATI
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 2. HYPERMODEL DINAMICO
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())
    
    # SCELTA DELL'ATTIVAZIONE: hp.Choice seleziona tra valori discreti in una lista
    # Usiamo la stessa attivazione per tutti i layer nascosti per coerenza
    activation_choice = hp.Choice('activation', values=['relu', 'tanh'])
    
    # NUMERO DI LAYER: hp.Int decide quanti layer densi aggiungere (da 1 a 3)
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(keras.layers.Dense(
            # Nota: usiamo un nome dinamico per i neuroni di ogni layer (units_0, units_1...)
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation=activation_choice
        ))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    # Tuning del Learning Rate (come nel codice precedente)
    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. INIZIALIZZAZIONE RANDOM SEARCH
# La Random Search esplora lo spazio in modo casuale, utile per trial limitati (10)
tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=10, # Budget di 10 tentativi
    directory='exercise_results',
    project_name='mnist_depth_vs_width'
)

# 4. ESECUZIONE DELLA RICERCA
tuner.search(x_train, y_train, epochs=3, validation_split=0.2)

# 5. ANALISI DEI RISULTATI
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n--- CONFIGURAZIONE VINCITRICE ---")
print(f"Funzione di attivazione preferita: {best_hps.get('activation')}")
print(f"Numero di layer densi: {best_hps.get('num_layers')}")

# Ciclo per stampare i neuroni di ogni layer trovato
for i in range(best_hps.get('num_layers')):
    print(f"Neuroni nel Layer {i}: {best_hps.get(f'units_{i}')}")

# Valutazione finale
best_model = tuner.get_best_models(num_models=1)[0]
_, accuracy = best_model.evaluate(x_test, y_test, verbose=0)
print(f"\nAccuratezza finale sul test set: {accuracy:.4f}")