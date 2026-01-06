import tensorflow as tf
import numpy as np

MODEL_PATH = "models/vgg16_model2.h5"

model = tf.keras.models.load_model(MODEL_PATH)
print("Model output shape:", model.output_shape)

CLASS_NAMES = ['Curl Virus', 'Healthy', 'Leaf Reddening', 'Leaf Spot Bacterial Blight']

def predict(image_tensor: np.ndarray):
    preds = model.predict(image_tensor)[0]

    predicted_index = int(np.argmax(preds))
    confidence = float(preds[predicted_index])

    probabilities = {
        CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
    }

    return {
        "class_name": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": probabilities,
    }
