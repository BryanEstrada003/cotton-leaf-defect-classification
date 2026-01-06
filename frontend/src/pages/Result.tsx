import { useEffect, useState } from 'react'
import type { PredictionResult } from '../types/prediction'

import DiagnosisCard from '../components/DiagnosisCard'
import ConfidenceBar from '../components/ConfidenceBar'
import ProbabilityChart from '../components/ProbabilityChart'
import RecommendationList from '../components/RecommendationList'
import GradCamOverlay from '../components/GradCamOverlay'
import Loader from '../components/Loader'

import heatmapMock from '../assets/heatmap_mock.png'

export default function Result() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [image, setImage] = useState<string | null>(null)

  useEffect(() => {
    const stored = localStorage.getItem('prediction')
    const img = localStorage.getItem('uploaded_image')

    if (stored) setResult(JSON.parse(stored))
    if (img) setImage(img)
  }, [])

  if (!result || !image) {
    return <Loader />
  }

  return (
    <>
      <DiagnosisCard result={result} />
      <ConfidenceBar confidence={result.confidence} />
      <ProbabilityChart probabilities={result.probabilities} />

      <GradCamOverlay
        image={image}
      />

      <RecommendationList recommendations={result.recommendations} />
    </>
  )
}
