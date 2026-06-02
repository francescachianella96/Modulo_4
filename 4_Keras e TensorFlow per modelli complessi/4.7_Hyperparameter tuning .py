import tensorflow as tf
import keras
import keras_tuner as kt #da installare a parte con pip install keras_tuner
import numpy as np

# 1. PREPARAZIONE DEI DATI
# Utilizziamo MNIST, normalizzando i pixel tra 0 e 1 per favorire la convergenza
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# 2. FUNZIONE DI COSTRUZIONE DEL MODELLO (Hypermodel)
# Questa funzione definisce lo "spazio di ricerca": quali parametri testare?
def build_model(hp):
    model = keras.Sequential([
        # Definiamo l'input in modo esplicito (Standard Keras 3)
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        
        # TUNING DEI NEURONI: hp.Int crea un range di interi
        # Il Tuner proverà valori come 32, 64, 96... fino a 256
        keras.layers.Dense(
            units=hp.Int('units', min_value=32, max_value=256, step=32),
            activation='relu'
        ),
        
        keras.layers.Dense(10, activation='softmax')
    ])
    
    # TUNING DEL LEARNING RATE: hp.Float con scala logaritmica
    # La scala logaritmica è ideale per il LR perché esplora ordini di grandezza diversi
    learning_rate = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. INIZIALIZZAZIONE DEL TUNER BAYESIANO
# A differenza della ricerca casuale, l'ottimizzazione Bayesiana usa la statistica 
# per prevedere quali combinazioni di parametri funzioneranno meglio basandosi sui risultati passati.
tuner = kt.BayesianOptimization(
    hypermodel=build_model,
    objective='val_accuracy', # Vogliamo massimizzare la precisione sul set di validazione
    max_trials=5,             # Numero massimo di modelli differenti da testare
    directory='tuning_results',
    project_name='mnist_keras_2025'
)

# 4. ESECUZIONE DELLA RICERCA (Search)
# Funziona esattamente come il .fit() di Keras. Il Tuner gestisce i cicli di addestramento.
print("--- Inizio ricerca iperparametri ---")
tuner.search(
    x_train, y_train, 
    epochs=3, 
    validation_split=0.2,
    verbose=1
)

# 5. ESTRAZIONE DEI RISULTATI MIGLIORI
# Recuperiamo i parametri che hanno ottenuto la 'val_accuracy' più alta
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("\n--- RISULTATI DELLA RICERCA ---")
print(f"Numero ottimale di neuroni: {best_hps.get('units')}")
print(f"Learning Rate ottimale: {best_hps.get('lr'):.5f}")

# Recuperiamo il modello "vincitore" già pronto per l'uso
best_model = tuner.get_best_models(num_models=1)[0]

# Valutazione finale sul test set
loss, accuracy = best_model.evaluate(x_test, y_test, verbose=0)
print(f"Accuratezza finale sul test set: {accuracy:.4f}")