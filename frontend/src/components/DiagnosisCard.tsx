import { Card, CardContent, Typography, Chip, Box } from '@mui/material'
import type { PredictionResult } from '../types/prediction'

interface Props {
  result: PredictionResult
}

export default function DiagnosisCard({ result }: Props) {
  return (
    <Card sx={{ mt: 4 }}>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Diagnóstico
        </Typography>

        <Typography variant="h6" color="primary">
          {result.class}
        </Typography>

        <Box sx={{ mt: 2 }}>
          <Chip
            label={`Modelo: ${result.model_used}`}
            sx={{ mr: 1 }}
          />
          <Chip
            label={`Tiempo: ${result.inference_time_ms} ms`}
            color="secondary"
          />
        </Box>
      </CardContent>
    </Card>
  )
}
