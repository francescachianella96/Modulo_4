import tensorflow as tf

# 1. Generazione Input
X = tf.random.normal([3, 4])

# 2. Definizione Pesi (Variabili)
W = tf.Variable(tf.random.normal([4, 2]))

# 3. Moltiplicazione Matriciale (Dot Product)
# Usiamo l'operatore @ che è zucchero sintattico per tf.matmul
output_intermedio = X @ W

# 4. Definizione Bias e Somma con Broadcasting
b = tf.Variable([0.5, -0.5]) # Vettore di dimensione 2
output_finale = output_intermedio + b

print(f"Shape di X: {X.shape}")
print(f"Shape di W: {W.shape}")
print(f"Shape Finale: {output_finale.shape}")

#SPIEGAZIONE:
# Perché funziona? 
# La moltiplicazione X(3,4) * W(4,2) è possibile perché la dimensione interna (4) coincide.
# Il risultato è (3,2). Il bias b ha dimensione (2). 
# TensorFlow vede che la dimensione finale (2) coincide con b e 'copia' b 
# per tre volte (una per ogni esempio del batch) per eseguire la somma.