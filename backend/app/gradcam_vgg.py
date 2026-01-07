import tensorflow as tf
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image


def generate_gradcam(
    model: tf.keras.Model,
    image_tensor: np.ndarray,
    class_index: int,
    conv_layer_name: str = "block5_conv3",
) -> str:
    """
    Genera Grad-CAM en base64 para un modelo VGG16 embebido.
    """

    # Extraer submodelo VGG16
    vgg = model.get_layer("vgg16")

    # Modelo Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            vgg.get_layer(conv_layer_name).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        loss = predictions[:, class_index]

    # Gradientes
    grads = tape.gradient(loss, conv_outputs)

    # Global Average Pooling sobre gradientes
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Ponderar mapas de activación
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    # Normalizar
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    # Redimensionar a tamaño original
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    # Aplicar colormap
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Convertir a base64
    image = Image.fromarray(heatmap)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"
