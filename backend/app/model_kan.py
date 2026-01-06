# app/model_kan.py
import torch
from pathlib import Path
from app.kan_model import Net

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "kan_model.pth"

device = torch.device("cpu")

model_kan = Net(num_classes=6)
state_dict = torch.load(MODEL_PATH, map_location=device)
model_kan.load_state_dict(state_dict)
model_kan.eval()

CLASS_NAMES = [
    "Alphids",
    "Army worm",
    "Bacterial Blight",
    "Healthy",
    "Powdery Mildew",
    "Target spot"
]


@torch.no_grad()
def predict_kan(image_tensor):
    outputs = model_kan(image_tensor)
    probs = torch.softmax(outputs, dim=1)[0]

    idx = probs.argmax().item()
    return {
        "class_name": CLASS_NAMES[idx],
        "confidence": probs[idx].item(),
        "probabilities": {
            CLASS_NAMES[i]: probs[i].item()
            for i in range(len(CLASS_NAMES))
        },
    }
