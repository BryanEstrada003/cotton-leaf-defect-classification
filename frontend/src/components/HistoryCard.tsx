import {
  Card,
  CardContent,
  CardMedia,
  Typography,
  Chip,
} from '@mui/material'
import { useNavigate } from 'react-router-dom'
import { useCurrentDiagnosisStore } from '../stores/useCurrentDiagnosisStore'

export default function HistoryCard({ item }: any) {
  const navigate = useNavigate()
  const setCurrent = useCurrentDiagnosisStore((s) => s.setCurrent)

  const handleClick = () => {
    setCurrent(item)
    navigate('/result')
  }

  const isCorrected = item.review?.status === 'corrected'

  const renderReviewChip = () => {
    if (!item.review) {
      return <Chip label="Pendiente" size="small" />
    }

    if (item.review.status === 'corrected') {
      return <Chip label="Corregido" color="warning" size="small" />
    }

    if (item.review.status === 'approved') {
      return <Chip label="Confirmado" color="success" size="small" />
    }

    return null
  }

  return (
    <Card onClick={handleClick} sx={{ cursor: 'pointer' }}>
      <CardMedia
        component="img"
        height="160"
        image={item.image}
        alt="Hoja analizada"
      />

      <CardContent>
        <Typography variant="subtitle1">
          {isCorrected
            ? item.review?.reviewed_class
            : item.result.class}
        </Typography>

        {isCorrected ? (
          <Typography variant="body2" color="text.secondary" noWrap>
            {item.review?.comments}
          </Typography>
        ) : (
          <Typography variant="body2" color="text.secondary">
            Confianza: {(item.result.confidence * 100).toFixed(1)}%
          </Typography>
        )}

        <div style={{ marginTop: 8 }}>{renderReviewChip()}</div>
        
        {!isCorrected ? (
        <Typography variant="caption" display="block" sx={{ mt: 1 }}>
          {new Date(item.createdAt).toLocaleString()}
        </Typography>
        ) : (
        <Typography variant="caption" display="block" sx={{ mt: 1 }}>
          {new Date(item.review?.reviewedAt).toLocaleString()}
        </Typography>
        )}
      </CardContent>
    </Card>
  )
}
