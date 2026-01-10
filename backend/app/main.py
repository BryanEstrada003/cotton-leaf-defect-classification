from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import time

from app.schemas import PredictionResponse

# ===== KAN (PyTorch) =====
from app.preprocessing_kan import load_and_preprocess_image_kan
from app.model import predict_kan_with_gradcam


# ------------------------
# App principal
# ------------------------
app = FastAPI(
    title="CotVision API",
    description="API para clasificación de enfermedades en hojas de algodón",
    version="1.0.0",
)

# ------------------------
# CORS
# ------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------
# Router con prefijo /api
# ------------------------
router = APIRouter(prefix="/api")

@router.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    image: UploadFile = File(...),
):
    """
    Endpoint de inferencia.
    Grad-CAM para KAN.
    """
    start_time = time.time()

    image_bytes = await image.read()

    image_tensor = load_and_preprocess_image_kan(image_bytes)
    result = predict_kan_with_gradcam(image_tensor)

    inference_time_ms = int((time.time() - start_time) * 1000)

    return {
        "class_name": result["class_name"],
        "confidence": result["confidence"],
        "inference_time_ms": inference_time_ms,
        "probabilities": result["probabilities"],
        "heatmap_url": result.get("heatmap_base64"),
    }

# ------------------------
# Registrar router
# ------------------------
app.include_router(router)
