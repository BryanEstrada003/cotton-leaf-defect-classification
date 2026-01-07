# app/model_kan.py
import torch
from pathlib import Path
from io import BytesIO
import base64
from PIL import Image

from app.kan_model import Net
from app.gradcam_kan import generate_gradcam_kan

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "kan_model2.pth"

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "Curl Virus",
    "Healthy",
    "Leaf Reddening",
    "Leaf Spot Bacterial Blight",
]

# 🔹 Cargar modelo
model_kan = Net(num_classes=len(CLASS_NAMES))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model_kan.load_state_dict(state_dict)
model_kan.to(DEVICE)
model_kan.eval()


@torch.no_grad()
def predict_kan(image_tensor: torch.Tensor):
    outputs = model_kan(image_tensor)
    probs = torch.softmax(outputs, dim=1)[0]

    idx = probs.argmax().item()

    return {
        "class_name": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        },
    }


def predict_kan_with_gradcam(image_tensor: torch.Tensor):
    image_tensor = image_tensor.to(DEVICE)

    outputs = model_kan(image_tensor)
    probs = torch.softmax(outputs, dim=1)[0]

    idx = probs.argmax().item()

    # 🔹 Grad-CAM (CNN + KAN, como en el paper)
    heatmap = generate_gradcam_kan(
        model_kan,
        image_tensor,
        class_idx=idx,
        target_size=(224, 224),
    )

    # 🔹 Convertir a base64 (frontend-friendly)
    img = Image.fromarray(heatmap)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "class_name": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        },
        "heatmap_base64": heatmap_b64,
    }
