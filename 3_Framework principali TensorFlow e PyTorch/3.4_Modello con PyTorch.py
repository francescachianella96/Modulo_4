import torch
import torch.nn as nn # per far funzionare i layer
import torch.nn.functional as F # per funzioni di attivazioni senza parametri

# 1. DEFINIZIONE DELLA CLASSE
# Ereditiamo da nn.Module per registrare automaticamente i parametri
class MioClassificatore(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # Chiamata obbligatoria al costruttore della classe base
        super(MioClassificatore, self).__init__()
        
        # 2. DEFINIZIONE DEI LAYER (COSTRUTTORE)
        # Primo layer lineare: trasforma input_size -> hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Secondo layer lineare: trasforma hidden_size -> num_classes (output)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Nota: I pesi e i bias vengono inizializzati automaticamente qui
        
    # 3. DEFINIZIONE DEL FLUSSO (FORWARD PASS)
    def forward(self, x):
        # Passaggio nel primo layer
        x = self.fc1(x)
        
        # Applicazione della non-linearità (ReLU)
        # Usiamo F.relu perché non ha pesi da addestrare
        x = F.relu(x)
        
        # Passaggio finale per ottenere i logit di output
        x = self.fc2(x)
        
        return x

# --- TEST DEL MODELLO ---
# Ipotizziamo: 10 feature in ingresso, 20 neuroni nascosti, 3 classi in uscita
input_dim = 10
hidden_dim = 20
output_dim = 3

model = MioClassificatore(input_dim, hidden_dim, output_dim)

# Creiamo un batch di dati finti (5 campioni, 10 feature ciascuno)
dummy_input = torch.randn(5, input_dim)

# Eseguiamo il forward pass chiamando l'oggetto direttamente
output = model(dummy_input)

print(f"Architettura del modello:\n{model}")
print(f"\nShape dell'output: {output.shape}") # Dovrebbe essere [5, 3]
print(f"I gradienti sono attivi? {output.requires_grad}")