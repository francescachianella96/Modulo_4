import numpy as np

def calculate_iou(boxA, boxB):
    # ... (La funzione rimane identica all'originale per brevità) ...
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

# --- NUOVA LOGICA: NON-MAXIMUM SUPPRESSION (NMS) ---

# 1. Definiamo i box e le loro confidenze (già ordinati per punteggio decrescente)
boxes = [
    [50, 50, 150, 150],  # Box A (Confidenza 0.9) - IL CAMPIONE
    [60, 60, 170, 160],  # Box B (Confidenza 0.7) - IL POSSIBILE DUPLICATO
    [200, 200, 300, 300] # Box C (Confidenza 0.4) - UN ALTRO OGGETTO
]
scores = [0.9, 0.7, 0.4]
threshold = 0.5 # Soglia IoU sopra la quale eliminiamo il box

final_boxes = []

# 2. Iteriamo finché ci sono box nella lista
while len(boxes) > 0:
    # Prendi il box con la confidenza più alta (il primo, dato che sono ordinati)
    current_best = boxes.pop(0)
    final_boxes.append(current_best)
    
    # 3. SFIDA: Confronta il "campione" con i restanti box
    # Se IoU > 0.5, eliminiamo il box dalla lista perché è un duplicato
    boxes = [box for box in boxes if calculate_iou(current_best, box) <= threshold]

# --- RISULTATO ---
print(f"Box rimasti dopo NMS: {len(final_boxes)}")
for i, box in enumerate(final_boxes):
    print(f"Box finale {i+1}: {box}")