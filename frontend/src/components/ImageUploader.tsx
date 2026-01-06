// components/ImageUploader.tsx
import { Button, Box } from '@mui/material'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { sendImage } from '../services/api'

interface Props {
  model: 'vgg16' | 'kan'
}

export default function ImageUploader({ model }: Props) {
  const [image, setImage] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async () => {
    if (!image) return

    setLoading(true)

    // Guardamos la imagen para Result.tsx
    const reader = new FileReader()
    reader.onload = () => {
      localStorage.setItem('uploaded_image', reader.result as string)
    }
    reader.readAsDataURL(image)

    console.log('Enviando modelo:', model)

    const result = await sendImage(image, model)
    localStorage.setItem('prediction', JSON.stringify(result))

    navigate('/result')
  }

  return (
    <Box>
      <input
        type="file"
        accept="image/*"
        onChange={(e) =>
          setImage(e.target.files ? e.target.files[0] : null)
        }
      />

      <Button
        variant="contained"
        sx={{ mt: 2 }}
        disabled={!image || loading}
        onClick={handleSubmit}
      >
        {loading ? 'Analizando...' : 'Analizar'}
      </Button>
    </Box>
  )
}
