import random

def predict_face(face):
    confidence = round(random.uniform(0.70, 0.99), 2)
    label = random.choice(["REAL", "FAKE"])

    return {
        "label": label,
        "confidence": confidence
    }