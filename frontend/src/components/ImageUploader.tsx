import { Button, Box } from '@mui/material'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { sendImage } from '../services/api'

export default function ImageUploader() {
const [image, setImage] = useState<File | null>(null)
const [loading, setLoading] = useState(false)
const navigate = useNavigate()

const handleSubmit = async () => {
  if (!image) return

  setLoading(true)

  const reader = new FileReader()
  reader.onload = () => {
    localStorage.setItem('uploaded_image', reader.result as string)
  }
  reader.readAsDataURL(image)

  const result = await sendImage(image)
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
