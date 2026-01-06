import {
  Card,
  CardContent,
  CardMedia,
  Typography,
  Chip,
} from '@mui/material'
import { useNavigate } from 'react-router-dom'

export default function HistoryCard({ item }: any) {
  const navigate = useNavigate()

  const handleClick = () => {
    localStorage.setItem('prediction', JSON.stringify(item.result))
    localStorage.setItem('uploaded_image', item.image)
    navigate('/result')
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
          {item.result.class}
        </Typography>

        <Typography variant="body2" color="text.secondary">
          Confianza: {(item.result.confidence * 100).toFixed(1)}%
        </Typography>

        <Typography variant="caption" display="block" sx={{ mt: 1 }}>
          {new Date(item.created_at).toLocaleString()}
        </Typography>

        <Chip
          label={item.model_used}
          size="small"
          sx={{ mt: 1 }}
        />
      </CardContent>
    </Card>
  )
}
