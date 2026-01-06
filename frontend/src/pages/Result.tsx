// pages/Result.tsx
import { useEffect, useState } from 'react'
import type { PredictionResult } from '../types/prediction'

import DiagnosisCard from '../components/DiagnosisCard'
import ConfidenceBar from '../components/ConfidenceBar'
import ProbabilityChart from '../components/ProbabilityChart'
import RecommendationList from '../components/RecommendationList'
import GradCamOverlay from '../components/GradCamOverlay'
import Loader from '../components/Loader'

import { recommendationsByClass } from '../utils/recommendations'

export default function Result() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [image, setImage] = useState<string | null>(null)

  useEffect(() => {
    const storedPrediction = localStorage.getItem('prediction')
    const storedImage = localStorage.getItem('uploaded_image')

    if (storedPrediction) {
      setResult(JSON.parse(storedPrediction))
    }

    if (storedImage) {
      setImage(storedImage)
    }
  }, [])

  if (!result || !image) {
    return <Loader />
  }

  const recommendations =
    recommendationsByClass[result.class] ?? [
      'No hay recomendaciones disponibles para esta clase.',
    ]

  return (
    <>
      <DiagnosisCard result={result} />

      <ConfidenceBar confidence={result.confidence} />

      <ProbabilityChart probabilities={result.probabilities} />

      <GradCamOverlay image={image} />

      <RecommendationList items={recommendations} />
    </>
  )
}
