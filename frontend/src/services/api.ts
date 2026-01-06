// services/api.ts
import axios from 'axios'
import type { PredictionResult } from '../types/prediction'

const API_URL = 'http://localhost:8000'

export async function sendImage(
  image: File,
  model: 'vgg16' | 'kan'
): Promise<PredictionResult> {
  const formData = new FormData()
  formData.append('image', image)
  formData.append('model', model) // 🔥 CLAVE

  const response = await axios.post(`${API_URL}/predict`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })

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
