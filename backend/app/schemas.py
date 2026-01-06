from typing import Dict, List, Optional
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    model_used: str
    inference_time_ms: int
    probabilities: Dict[str, float]
    heatmap_url: Optional[str]
    recommendations: List[str]
