# app/preprocessing_kan.py
from PIL import Image
from io import BytesIO
import torch
from torchvision import transforms

# Transformaciones EXACTAS usadas en entrenamiento
transform_kan = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

def load_and_preprocess_image_kan(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform_kan(image)
    tensor = tensor.unsqueeze(0)  # batch dimension
    return tensor
