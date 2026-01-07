from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import time

from app.schemas import PredictionResponse

# ===== VGG16 (TensorFlow) =====
from app.preprocessing import load_and_preprocess_image
from app.model import predict_with_cam as predict_vgg_with_cam


# ===== KAN (PyTorch) =====
from app.preprocessing_kan import load_and_preprocess_image_kan
from app.model_kan import predict_kan, predict_kan_with_gradcam


app = FastAPI(
    title="CotVision API",
    description="API para clasificación de enfermedades en hojas de algodón",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    Grad-CAM para VGG16 y KAN.
    """
    start_time = time.time()
    image_bytes = await image.read()

    # ===== SELECCIÓN DE MODELO =====
    if model.lower() == "kan":
        image_tensor = load_and_preprocess_image_kan(image_bytes)
        result = predict_kan_with_gradcam(image_tensor)
        model_used = "KAN"
        heatmap = result.get("heatmap_base64")

    else:
        image_tensor = load_and_preprocess_image(image_bytes)
        result = predict_vgg_with_cam(image_tensor)
        model_used = "VGG16"
        heatmap = result.get("heatmap_base64")

    inference_time_ms = int((time.time() - start_time) * 1000)

    return {
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "model_used": model_used,
        "inference_time_ms": inference_time_ms,
        "probabilities": result["probabilities"],
        "heatmap_url": heatmap,
        "recommendations": [
            "Monitorear la evolución de la hoja",
            "Aplicar tratamiento localizado si es necesario",
        ],
    }
