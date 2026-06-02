import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers

# 1. CARICAMENTO DEL MODELLO BASE
vgg_base = applications.VGG16(weights='imagenet', 
                              include_top=False, 
                              input_shape=(224, 224, 3))

# 2. SBLOCCO E CONGELAMENTO SELETTIVO (Richiesta esercizio)
# Per prima cosa sblocchiamo l'intera base
vgg_base.trainable = True

# Definiamo quanti layer vogliamo lasciare sbloccati alla fine della rete
fine_tune_at = len(vgg_base.layers) - 4

# Congeliamo tutti i layer fino a fine_tune_at
for layer in vgg_base.layers[:fine_tune_at]:
    layer.trainable = False

# 

# 3. COSTRUZIONE DEL MODELLO COMPLESSIVO
model = models.Sequential([
    vgg_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 4. COMPILAZIONE CON LEARNING RATE RIDOTTO
# Usiamo 1e-5 (0.00001) per aggiornare i pesi pre-addestrati con estrema cautela
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. ISPEZIONE DEI PARAMETRI
# Noterai che i "Trainable params" sono aumentati rispetto al congelamento totale
model.summary()

# Verifica visiva dei layer sbloccati
print(f"\nNumero totale di layer in VGG16: {len(vgg_base.layers)}")
print(f"Layer congelati: {fine_tune_at}")
print(f"Layer sbloccati per il fine-tuning: {len(vgg_base.layers) - fine_tune_at}")

for i, layer in enumerate(vgg_base.layers):
    print(f"Layer {i}: {layer.name} | Addestrabile: {layer.trainable}")