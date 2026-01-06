// pages/Analyze.tsx
import { Typography, Card, CardContent } from '@mui/material'
import { useState } from 'react'
import ImageUploader from '../components/ImageUploader'
import ModelToggle from '../components/ModelToggle'

export default function Analyze() {
  const [model, setModel] = useState<'vgg16' | 'kan'>('vgg16')

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
          <ModelToggle
            model={model}
            onChange={(newModel) => {
              setModel(newModel)
              console.log('Modelo seleccionado:', newModel)
            }}
          />

          <ImageUploader model={model} />
        </CardContent>
      </Card>
    </>
  )
}
