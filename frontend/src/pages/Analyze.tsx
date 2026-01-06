import { Typography, Card, CardContent } from '@mui/material'
import ImageUploader from '../components/ImageUploader'
import ModelToggle from '../components/ModelToggle'

export default function Analyze() {
  return (
    <>
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
          <ModelToggle
            value="vgg16"
            onChange={(model) =>
              console.log('Modelo seleccionado:', model)
            }
          />
        </CardContent>
      </Card>
    </>
  )
}
