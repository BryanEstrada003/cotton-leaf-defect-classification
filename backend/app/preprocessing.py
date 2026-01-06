import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.imagenet_utils import preprocess_input

INPUT_SIZE = (224, 224)  # AJUSTA si usaste otro

def tf_pytorch_style_preprocess(img: np.ndarray) -> np.ndarray:
    return preprocess_input(img, mode="torch")


def load_and_preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(INPUT_SIZE)

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = tf_pytorch_style_preprocess(img_array)

    return img_array
