// components/ImageUploader.tsx
import { Button, Box } from '@mui/material'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { sendImage } from '../services/api'
import { v4 as uuidv4 } from 'uuid'

export default function ImageUploader({ model }: { model: 'vgg16' | 'kan' }) {
  const [image, setImage] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleSubmit = async () => {
    if (!image) return
    setLoading(true)

    const reader = new FileReader()

    reader.onload = async () => {
      const base64Image = reader.result as string

      const result = await sendImage(image, model)

      // Guardar resultado actual
      localStorage.setItem('prediction', JSON.stringify(result))
      localStorage.setItem('uploaded_image', base64Image)

      // 🔥 Guardar en historial
      const historyRaw = localStorage.getItem('history')
      const history = historyRaw ? JSON.parse(historyRaw) : []

      history.unshift({
        id: uuidv4(),
        image: base64Image,
        result,
        model_used: result.model_used,
        created_at: new Date().toISOString(),
      })

      localStorage.setItem('history', JSON.stringify(history))

      navigate('/result')
    }

    reader.readAsDataURL(image)
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
