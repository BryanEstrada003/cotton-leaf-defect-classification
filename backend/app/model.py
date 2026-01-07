import tensorflow as tf
import numpy as np
import cv2
import base64

MODEL_PATH = "models/vgg16_model2.h5"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = [
    "Curl Virus",
    "Healthy",
    "Leaf Reddening",
    "Leaf Spot Bacterial Blight",
]

# -------------------------------------------------
# CAM PROXY (SIN GRADIENTES)
# -------------------------------------------------
def generate_cam_proxy(image_tensor: np.ndarray):
    """
    Proxy CAM usando activaciones de la última capa conv.
    Compatible con tu modelo actual.
    """

    # Extraer el backbone VGG16
    backbone = model.get_layer("vgg16")

    # Forward SOLO hasta conv
    conv_features = backbone(image_tensor, training=False)
    conv_features = conv_features[0]  # (7, 7, 512)

    # Promedio por canal
    cam = tf.reduce_mean(conv_features, axis=-1)

    cam = cam.numpy()
    cam = np.maximum(cam, 0)
    cam /= cam.max() + 1e-8

    # Resize a 224x224
    cam = cv2.resize(cam, (224, 224))

    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

    _, buffer = cv2.imencode(".png", heatmap)
    return base64.b64encode(buffer).decode("utf-8")


# -------------------------------------------------
# PREDICT + CAM
# -------------------------------------------------
def predict_with_cam(image_tensor: np.ndarray):
    preds = model(image_tensor, training=False)[0]

    idx = int(tf.argmax(preds))
    confidence = float(preds[idx])

    heatmap_base64 = generate_cam_proxy(image_tensor)

    return {
        "class_name": CLASS_NAMES[idx],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))
        },
        "heatmap_base64": heatmap_base64,
    }
