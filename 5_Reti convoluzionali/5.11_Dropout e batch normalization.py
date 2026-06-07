import keras
from keras import layers, models

def build_robust_cnn_2025(input_shape=(32, 32, 3)):
    """
    Costruisce una CNN ottimizzata con tecniche di regolarizzazione avanzata.
    Ideale per dataset come CIFAR-10 o MNIST (previo resize).
    """
    model = models.Sequential(name="Robust_CNN_2025")

    # 2. DEFINIZIONE ESPLICITA DELL'INPUT
    # A differenza del passato, definire l'Input layer è best practice per:
    # - Validazione immediata dei tensori (Eager Execution).
    # - Debugging della forma (shape) prima della compilazione.
    model.add(layers.Input(shape=input_shape))

    # --- BLOCCO 1: ESTRAZIONE FEATURE E STABILIZZAZIONE ---
    # use_bias=False: Quando segue una BatchNormalization (BN), il parametro 'bias' 
    # della convoluzione è matematicamente ridondante. La BN applica un termine 
    # di 'beta' (offset) che svolge la stessa funzione. Rimuoverlo riduce i parametri totali.
    model.add(layers.Conv2D(32, (3, 3), padding='same', use_bias=False))
    
    # BatchNormalization: Fondamentale per combattere l'Internal Covariate Shift.
    # Normalizza l'output del layer precedente, accelerando la convergenza 
    # e agendo come una leggera forma di regolarizzazione.
    model.add(layers.BatchNormalization()) 
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # --- BLOCCO 2: REGOLARIZZAZIONE SPAZIALE ---
    model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # SpatialDropout2D vs Dropout standard:
    # Nelle CNN, i pixel adiacenti sono fortemente correlati. Il Dropout standard 
    # spegne pixel casuali, ma l'informazione "sopravvive" tramite i vicini. 
    # SpatialDropout spegne interi canali (feature maps), costringendo la rete 
    # a non fare affidamento su specifici filtri per riconoscere un pattern.
    model.add(layers.SpatialDropout2D(0.3)) 

    # --- TESTA DEL MODELLO (CLASSIFICATORE) ---
    # Flatten trasforma il tensore 3D in un vettore 1D per i layer densi.
    model.add(layers.Flatten())
    
    # Dense Layer con Dropout: Qui usiamo il Dropout standard (0.5).
    # È la difesa finale contro l'overfitting nei layer fully-connected, 
    # dove risiede la maggior parte dei parametri addestrabili.
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5)) 
    
    # Output Layer: Softmax garantisce che la somma delle probabilità delle 10 classi sia 1.0.
    model.add(layers.Dense(10, activation='softmax'))

    return model

# Istanziamo il modello
model = build_robust_cnn_2025()

# 3. ISPEZIONE DEL SUMMARY
# Nota: Nel summary vedrai "Non-trainable params" nei layer di BatchNormalization.
# Questi rappresentano la 'media' e la 'varianza' mobile calcolate durante il training,
# che verranno "congelate" e usate durante l'inferenza (predizione).
model.summary()