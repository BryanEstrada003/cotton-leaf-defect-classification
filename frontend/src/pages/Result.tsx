import { useState } from 'react'
import { Box } from '@mui/material'

import DiagnosisCard from '../components/DiagnosisCard'
import ConfidenceBar from '../components/ConfidenceBar'
import ProbabilityChart from '../components/ProbabilityChart'
import GradCamOverlay from '../components/GradCamOverlay'
import Loader from '../components/Loader'
import DiagnosisReview from '../components/DiagnosisReview'
import ReviewCard from '../components/ReviewCard'
import { useNavigate } from 'react-router-dom'

import { useCurrentDiagnosisStore } from '../stores/useCurrentDiagnosisStore'
import { useHistoryStore } from '../stores/useHistoryStore'

export default function Result() {
  const current = useCurrentDiagnosisStore((s) => s.current)
  const updateDiagnosis = useHistoryStore((s) => s.updateDiagnosis)

  const [editingReview, setEditingReview] = useState(false)
  const navigate = useNavigate()


  if (!current) {
    return <Loader />
  }

  const { result, image, review } = current
  const isPinned = !!review

  const handleSaveReview = (reviewData: any) => {
    console.log('Saving review:', reviewData)
    updateDiagnosis(current.id, {
      review: reviewData,
    })
    console.log('Review saved.')
    setEditingReview(false)
    navigate('/history')

    console.log('Exited editing mode.')
  }

  const ReviewComponent =
    !review || editingReview ? (
      <DiagnosisReview
        detectedClass={result.class}
        initialReview={review}
        onSave={handleSaveReview}
      />
    ) : (
      <ReviewCard
        review={review}
        onEdit={() => setEditingReview(true)}
      />
    )

  return (
    <>
      <DiagnosisCard result={result} />

      {isPinned && ReviewComponent}

      <ConfidenceBar confidence={result.confidence} />
      <ProbabilityChart probabilities={result.probabilities} />

      <Box align="center">
        <GradCamOverlay
          image={image}
          heatmap={
            result.heatmap_url
              ? result.heatmap_url.startsWith('data:image')
                ? result.heatmap_url
                : `data:image/png;base64,${result.heatmap_url}`
              : null
          }
        />
      </Box>

      {!isPinned && ReviewComponent}
    </>
  )
}
