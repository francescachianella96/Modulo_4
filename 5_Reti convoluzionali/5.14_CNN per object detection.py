import numpy as np

def calculate_iou(boxA, boxB):
    """
    Calcola l'Intersection over Union (IoU) tra due bounding box.
    
    Formato input atteso: [x1, y1, x2, y2]
    (x1, y1) -> Coordinate del vertice in alto a sinistra (Top-Left)
    (x2, y2) -> Coordinate del vertice in basso a destra (Bottom-Right)
    """
    
    # 1. DETERMINAZIONE DEL RETTANGOLO INTERSEZIONE
    # L'intersezione di due rettangoli è essa stessa un rettangolo.
    # Per trovare il vertice in alto a sinistra dell'intersezione, prendiamo il MASSIMO tra le x1 e le y1.
    # Per trovare il vertice in basso a destra, prendiamo il MINIMO tra le x2 e le y2.
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 2. CALCOLO DELL'AREA DI INTERSEZIONE
    # Calcoliamo larghezza (xB - xA) e altezza (yB - yA) dell'area comune.
    # Aggiungiamo "+ 1" perché le coordinate dei pixel sono inclusive (es. da 0 a 9 sono 10 pixel).
    # Il max(0, ...) è fondamentale: se i box non si sovrappongono, la sottrazione darebbe 
    # un numero negativo. Forzando a 0, l'area di intersezione sarà correttamente nulla.
    interWidth = max(0, xB - xA + 1)
    interHeight = max(0, yB - yA + 1)
    interArea = interWidth * interHeight

    # 3. CALCOLO DELLE AREE DEI SINGOLI BOX
    # Calcoliamo l'area occupata da ciascun rettangolo separatamente.
    # Usiamo la stessa logica (lato + 1) per coerenza con il calcolo dell'intersezione.
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # 4. CALCOLO DELL'UNIONE E DELL'IOU
    # L'Unione non è semplicemente AreaA + AreaB, perché l'intersezione verrebbe contata due volte.
    # La formula corretta è: Area(A ∪ B) = Area(A) + Area(B) - Area(A ∩ B)
    unionArea = float(boxAArea + boxBArea - interArea)
    
    # L'IoU è il rapporto tra quanto i box "condividono" e quanto spazio "occupano insieme".
    # Il valore è sempre compreso tra 0 (nessuna sovrapposizione) e 1 (sovrapposizione perfetta).
    iou = interArea / unionArea

    return iou

# --- TEST E ANALISI ---
# Ground truth: la posizione corretta dell'oggetto nel dataset
ground_truth = [50, 50, 150, 150]
# Prediction: la scatola disegnata dal nostro modello di Deep Learning
prediction = [60, 60, 170, 160]

result = calculate_iou(ground_truth, prediction)
print(f"Intersection over Union: {result:.4f}")