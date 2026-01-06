# app/model_kan.py
import torch
from pathlib import Path
from app.kan_model import Net

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "kan_model.pth"

DEVICE = torch.device("cpu")

CLASS_NAMES = ['Curl Virus', 'Healthy', 'Leaf Reddening', 'Leaf Spot Bacterial Blight']

model_kan = Net(num_classes=len(CLASS_NAMES))
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model_kan.load_state_dict(state_dict)
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
