export interface PredictionResult {
  class: string
  confidence: number
  model_used: 'KAN' | 'VGG16'
  inference_time_ms: number
  probabilities: Record<string, number>
  heatmap_url: string | null
  recommendations: string[]
}
