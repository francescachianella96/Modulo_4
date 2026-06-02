import os
import time
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

# 1. GENERAZIONE DATI SINTETICI (Standard per entrambi)
num_samples = 10000
input_dim = 10
X_np = np.random.randn(num_samples, input_dim).astype(np.float32)
y_np = np.random.randint(0, 2, size=(num_samples,)).astype(np.int64)

# --- VERSIONE TENSORFLOW / KERAS ---
def build_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

tf_model = build_keras_model()
tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Conteggio parametri TF
tf_params = tf_model.count_params()

# --- VERSIONE PYTORCH ---
class PTModel(nn.Module):
    def __init__(self):
        super(PTModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

pt_model = PTModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pt_model.parameters())

# Conteggio parametri PT
pt_params = sum(p.numel() for p in pt_model.parameters())

# --- COMPARAZIONE E TRAINING TIME ---
print(f"Parametri TensorFlow: {tf_params}")
print(f"Parametri PyTorch:    {pt_params}")

# Test Tempo TF
start = time.time()
tf_model.fit(X_np, y_np, epochs=1, batch_size=32, verbose=0)
print(f"Tempo Epoca TF: {time.time() - start:.4f} secondi")

# Test Tempo PT
X_pt = torch.tensor(X_np)
y_pt = torch.tensor(y_np)
start = time.time()
for i in range(0, num_samples, 32):
    optimizer.zero_grad()
    outputs = pt_model(X_pt[i:i+32])
    loss = criterion(outputs, y_pt[i:i+32])
    loss.backward()
    optimizer.step()
print(f"Tempo Epoca PT: {time.time() - start:.4f} secondi")