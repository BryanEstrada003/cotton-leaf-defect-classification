import random
import time


def predict_mock():
    start = time.time()

    classes = [
        "Bacterial Blight",
        "Powdery Mildew",
        "Aphids",
        "Healthy",
    ]

    probs = [random.random() for _ in classes]
    total = sum(probs)
    probs = [p / total for p in probs]

    prediction = dict(zip(classes, probs))
    best_class = max(prediction, key=prediction.get)

    time.sleep(0.8)  # simula inferencia pesada

    return {
        "class_name": best_class,
        "confidence": prediction[best_class],
        "model_used": "KAN",
        "inference_time_ms": int((time.time() - start) * 1000),
        "probabilities": prediction,
        "recommendations": [
            "Monitorear la evolución de la hoja",
            "Aplicar tratamiento localizado si persiste",
        ],
    }
