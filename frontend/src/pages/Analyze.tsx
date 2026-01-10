// pages/Analyze.tsx
import { Typography, Card, CardContent } from '@mui/material'
import ImageUploader from '../components/ImageUploader'
import { Box } from '@mui/material'

export default function Analyze() {

  return (
    <Box align="center">
      <Typography variant="h4" gutterBottom>
        Análisis de imagen
      </Typography>

      <Typography color="text.secondary" sx={{ mb: 3 }}>
        Sube una imagen clara de una hoja de algodón para obtener un diagnóstico
        asistido por inteligencia artificial.
      </Typography>

      <Card>
        <CardContent>
          <ImageUploader />
        </CardContent>
      </Card>
    </Box>
  )
}
