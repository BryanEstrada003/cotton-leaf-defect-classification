import {
  Card,
  CardContent,
  CardMedia,
  Typography,
  Chip
} from '@mui/material'
import { useNavigate } from 'react-router-dom'

export default function HistoryCard({ item }: any) {
  const navigate = useNavigate()

  const handleClick = () => {
    localStorage.setItem('prediction', JSON.stringify(item.result))
    localStorage.setItem('uploaded_image', item.image)
    navigate('/result')
  }

  const isCorrected = item.review?.status === 'corrected'

  const renderReviewChip = () => {
    // ⏳ Pendiente
    if (!item.review) {
      return <Chip label="Pendiente" color="default" size="small" />
    }

    // ✏️ Corregido
    if (item.review.status === 'corrected') {
      return <Chip label="Corregido" color="warning" size="small" />
    }

    // ✅ Confirmado
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
          <Typography
            variant="body2"
            color="text.secondary"
            noWrap
            sx={{
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}
          >
            {item.review?.comments}
          </Typography>
        ) : (
          <Typography variant="body2" color="text.secondary">
            Confianza: {(item.result.confidence * 100).toFixed(1)}%
          </Typography>
        )}

        <div style={{ marginTop: 8 }}>
          {renderReviewChip()}
        </div>

        <Typography variant="caption" display="block" sx={{ mt: 1 }}>
          {new Date(item.created_at).toLocaleString()}
        </Typography>
      </CardContent>
    </Card>
  )
}
