import axios from 'axios'
import type { PredictionResult } from '../types/prediction'

const API_URL = 'http://localhost:8000'

export async function sendImage(
  image: File
): Promise<PredictionResult> {
  const formData = new FormData()
  formData.append('image', image)

  const response = await axios.post(`${API_URL}/predict`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

  // Adaptamos el contrato del backend al frontend
  return {
    class: response.data.class_name,
    confidence: response.data.confidence,
    model_used: response.data.model_used,
    inference_time_ms: response.data.inference_time_ms,
    probabilities: response.data.probabilities,
    heatmap_url: response.data.heatmap_url,
    recommendations: response.data.recommendations,
  }
}


