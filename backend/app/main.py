from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import time

from app.schemas import PredictionResponse

# ===== VGG16 (TensorFlow) =====
from app.preprocessing import load_and_preprocess_image
from app.model import predict as predict_vgg

# ===== KAN (PyTorch) =====
from app.preprocessing_kan import load_and_preprocess_image_kan
from app.model_kan import predict_kan


app = FastAPI(
    title="CotVision API",
    description="API para clasificación de enfermedades en hojas de algodón",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción se restringe
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    image: UploadFile = File(...),
    model: str = Form("vgg16"),  # "vgg16" | "kan"
):
    """
    Endpoint de inferencia.
    Permite seleccionar el modelo de clasificación (VGG16 o KAN).
    """
    start_time = time.time()

    # Leer imagen
    image_bytes = await image.read()

    # Selección de modelo
    if model.lower() == "kan":
        image_tensor = load_and_preprocess_image_kan(image_bytes)
        result = predict_kan(image_tensor)
        model_used = "KAN"
    else:
        image_tensor = load_and_preprocess_image(image_bytes)
        result = predict_vgg(image_tensor)
        model_used = "VGG16"

    inference_time_ms = int((time.time() - start_time) * 1000)

    return {
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "model_used": model_used,
        "inference_time_ms": inference_time_ms,
        "probabilities": result["probabilities"],
        "heatmap_url": None,  # Grad-CAM solo se aplicará a VGG16
        "recommendations": [
            "Monitorear la evolución de la hoja",
            "Aplicar tratamiento localizado si es necesario",
        ],
    }
