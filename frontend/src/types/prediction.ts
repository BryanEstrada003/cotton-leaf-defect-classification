export interface PredictionResult {
  class: string
  confidence: number
  inference_time_ms: number
  probabilities: Record<string, number>
  heatmap_url: string | null
}
