import {
  Box,
  Button,
  TextField,
  Typography,
  MenuItem,
  Card,
  CardContent,
} from '@mui/material'
import { useState, useEffect } from 'react'

const ALL_CLASSES = [
  'Curl Virus',
  'Healthy',
  'Leaf Reddening',
  'Leaf Spot Bacterial Blight',
]

interface Props {
  detectedClass: string
  initialReview?: any
  onSave: (review: any) => void
}

export default function DiagnosisReview({
  detectedClass,
  initialReview,
  onSave,
}: Props) {
  const [status, setStatus] = useState<'approved' | 'corrected'>('approved')
  const [reviewedClass, setReviewedClass] = useState('')
  const [comments, setComments] = useState('')

  useEffect(() => {
    if (initialReview) {
      setStatus(initialReview.status)
      setReviewedClass(
        initialReview.status === 'corrected'
          ? initialReview.reviewed_class
          : ''
      )
      setComments(initialReview.comments || '')
    }
  }, [initialReview])

  const availableClasses = ALL_CLASSES.filter(
    (c) => c !== detectedClass
  )

  const handleSave = () => {
    onSave({
      status,
      reviewed_class:
        status === 'approved' ? detectedClass : reviewedClass,
      comments,
      reviewed_at: new Date().toISOString(),
    })
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Revisión profesional
        </Typography>

        <TextField
          select
          fullWidth
          label="Estado del diagnóstico"
          value={status}
          onChange={(e) =>
            setStatus(e.target.value as 'approved' | 'corrected')
          }
          sx={{ mb: 2 }}
        >
          <MenuItem value="approved">Aprobado</MenuItem>
          <MenuItem value="corrected">Corregido</MenuItem>
        </TextField>

        {status === 'corrected' && (
          <TextField
            select
            fullWidth
            label="Clase corregida"
            value={reviewedClass}
            onChange={(e) => setReviewedClass(e.target.value)}
            sx={{ mb: 2 }}
          >
            {availableClasses.map((cls) => (
              <MenuItem key={cls} value={cls}>
                {cls}
              </MenuItem>
            ))}
          </TextField>
        )}

        <TextField
          fullWidth
          label="Comentarios"
          multiline
          rows={3}
          value={comments}
          onChange={(e) => setComments(e.target.value)}
          sx={{ mb: 2 }}
        />

        <Button
          variant="contained"
          onClick={handleSave}
          disabled={status === 'corrected' && !reviewedClass}
        >
          Guardar revisión
        </Button>
      </CardContent>
    </Card>
  )
}
