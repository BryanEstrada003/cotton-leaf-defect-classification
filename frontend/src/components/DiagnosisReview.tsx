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
  'Hoja Sana',
  'Leaf Reddening',
  'Bacterial Blight',
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
      reviewedAt: new Date().toISOString(),
    })
  }

  return (
    <Card sx={{ mt: 2 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Revisión profesional
        </Typography>

        <TextField
          select
          fullWidth
          label="Decisión"
          value={status}
          onChange={(e) =>
            setStatus(e.target.value as 'approved' | 'corrected')
          }
          sx={{ mb: 2 }}
        >
          <MenuItem value="approved">Aprobar</MenuItem>
          <MenuItem value="corrected">Corregir</MenuItem>
        </TextField>

        {status === 'corrected' && (
          <TextField
            select
            fullWidth
            label="Nuevo diagnóstico"
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
