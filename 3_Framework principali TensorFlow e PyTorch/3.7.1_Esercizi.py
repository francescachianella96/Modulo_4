import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAZIONE AMBIENTE ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    # 1. PREPARAZIONE DATI
    print("Caricamento dati California Housing...")
    # Carichiamo il dataset (8 caratteristiche numeriche)
    housing = fetch_california_housing()
    x, y = housing.data, housing.target

    # Suddivisione in Training e Test set
    x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

    # NORMALIZZAZIONE: Fondamentale per dati tabellari con scale diverse
    # (es. popolazione vs numero di stanze)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    # 2. DEFINIZIONE DEL MODELLO
    model = models.Sequential([
        # L'input ora ha 8 caratteristiche (features)
        Input(shape=(x_train.shape[1],), name="Input_Layer"),
        
        # Non serve Flatten() perché i dati sono già vettori monodimensionali
        layers.Dense(64, activation='relu', name="Hidden_Layer_1"),
        layers.Dense(64, activation='relu', name="Hidden_Layer_2"),
        layers.Dropout(0.1, name="Regularization"),
        
        # OUTPUT: 1 solo neurone senza attivazione (o 'linear') 
        # perché dobbiamo predire un valore continuo (prezzo), non una classe
        layers.Dense(1, name="Output_Layer")
    ])

    # COMPILAZIONE: Usiamo Mean Squared Error (MSE) per la regressione
    model.compile(
        optimizer='adam',
        loss='mean_squared_error', 
        metrics=['mean_absolute_error'] # MAE ci dice di quanti dollari sbagliamo in media
    )

    # 3. CONFIGURAZIONE TENSORBOARD
    log_dir = os.path.join(os.getcwd(), "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        write_graph=True,
        write_images=False # Disattivato: qui non abbiamo "immagini" dei pesi utili
    )

    # 4. TRAINING
    print(f"I log verranno salvati in: {log_dir}")
    model.fit(
        x_train, y_train, 
        epochs=30, # La regressione può richiedere più epoche per stabilizzarsi
        validation_data=(x_val, y_val), 
        callbacks=[tensorboard_callback],
        batch_size=32
    )

    # Valutazione finale
    test_loss = model.evaluate(x_test, y_test)
    print(f"\nLoss finale sul Test Set: {test_loss}")

if __name__ == "__main__":
    main()