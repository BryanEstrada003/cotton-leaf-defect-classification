import { Button, Box, Typography } from '@mui/material'
import ImageIcon from '@mui/icons-material/Image'
import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { sendImage } from '../services/api'
import { v4 as uuidv4 } from 'uuid'
import { useHistoryStore } from '../stores/useHistoryStore'
import { useCurrentDiagnosisStore } from '../stores/useCurrentDiagnosisStore'
import LoadingOverlay from './LoadingOverlay'

export default function ImageUploader() {
  const [image, setImage] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)

  const fileInputRef = useRef<HTMLInputElement | null>(null)

  const navigate = useNavigate()
  const addDiagnosis = useHistoryStore((s) => s.addDiagnosis)
  const setCurrent = useCurrentDiagnosisStore((s) => s.setCurrent)

  const handleSubmit = async () => {
    if (!image) return
    setLoading(true)

    const reader = new FileReader()

    reader.onload = async () => {
      const base64Image = reader.result as string
      const result = await sendImage(image)

      const diagnosis = {
        id: uuidv4(),
        image: base64Image,
        result,
        createdAt: new Date().toISOString(),
        review: undefined,
      }

      addDiagnosis(diagnosis)
      setCurrent(diagnosis)

      navigate('/result')
    }

    reader.readAsDataURL(image)
  }

  return (
    <>
      <Box
        sx={{
          minHeight: '20vh',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          gap: 3,
        }}
      >
        {/* INPUT OCULTO */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) =>
            setImage(e.target.files ? e.target.files[0] : null)
          }
        />

        {/* RECTÁNGULO DE SUBIDA */}
        <Box
          onClick={() => fileInputRef.current?.click()}
          sx={{
            width: 320,
            height: 220,
            border: '2px dashed #aaa',
            borderRadius: 3,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
            cursor: 'pointer',
            bgcolor: image ? 'rgba(0,0,0,0.03)' : 'transparent',
            transition: '0.2s',
            '&:hover': {
              borderColor: 'primary.main',
              bgcolor: 'rgba(0,0,0,0.04)',
            },
          }}
        >
          <ImageIcon sx={{ fontSize: 60, mb: 1, color: 'text.secondary' }} />
          <Typography variant="body1" color="text.secondary">
            {image ? image.name : 'Subir aquí imagen'}
          </Typography>
        </Box>

        {/* BOTÓN ANALIZAR */}
        <Button
          variant="contained"
          disabled={!image || loading}
          onClick={handleSubmit}
          sx={{ minWidth: 180 }}
        >
          {loading ? 'Analizando...' : 'Analizar'}
        </Button>
      </Box>

      <LoadingOverlay open={loading} />
    </>
  )
}
