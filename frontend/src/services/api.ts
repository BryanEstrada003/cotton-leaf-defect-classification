import axios from 'axios'
import type { PredictionResult } from '../types/prediction'

const api = axios.create({
  baseURL: '/api',
})

export async function sendImage(image: File): Promise<PredictionResult> {
  const formData = new FormData()
  formData.append('image', image)

  const response = await api.post('/predict', formData)

  return {
    class: response.data.class_name,
    confidence: response.data.confidence,
    inference_time_ms: response.data.inference_time_ms,
    probabilities: response.data.probabilities,
    heatmap_url: response.data.heatmap_url,
  }
}
