import numpy as np
import tensorflow as tf

#1. COSTANTI E VARIABILI: IL CUORE DELLO STATO
# Le costanti sono immutabili (input, iperparametri)
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])

# Le variabili sono mutabili (pesi, bias)
# Senza tf.Variable, l'ottimizzatore non saprebbe cosa aggiornare!
w = tf.Variable(tf.random.normal([2, 1]), name="weights")
b = tf.Variable(tf.zeros([1]), name="bias")

print(f"Costante:\n{a}")
print(f"Variabile Iniziale:\n{w.numpy()}\n")

# Modifica di una variabile: non usiamo '=', usiamo '.assign()'
w.assign(w * 2.0)
print(f"Variabile dopo raddoppio:\n{w.numpy()}\n")

# 2. OPERAZIONI VETTORIALI E BROADCASTING
# Calcolo element-wise accelerato
x = tf.constant([10, 20, 30], dtype=tf.float32)
y = tf.constant([1, 2, 3], dtype=tf.float32)

somma = tf.add(x, y) # Oppure x + y
prodotto_scalare = x * 5.0 # Broadcasting: il 5.0 viene espanso a [5, 5, 5]

print(f"Somma vettoriale: {somma}")
print(f"Broadcasting prodotto: {prodotto_scalare}\n")

# 3. IL MOTORE DELLE ANN: tf.matmul
# Creiamo un piccolo batch di input (3 esempi, 2 feature ciascuno)
inputs = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Moltiplicazione matriciale: (3x2) @ (2x1) = (3x1)
# Questo è il calcolo z = xW + b
z = tf.matmul(inputs, w) + b

print(f"Risultato Forward Pass (z):\n{z}")