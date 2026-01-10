import { Box, Typography, IconButton, Tooltip } from '@mui/material'
import { useState } from 'react'
import VisibilityIcon from '@mui/icons-material/Visibility'
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff'

interface Props {
  image: string
  heatmap: string
}

export default function GradCamOverlay({ image, heatmap }: Props) {
  const [showGradCam, setShowGradCam] = useState(true)

  return (
    <>
      <Typography variant="h6" sx={{ mt: 6, mb: 2 }}>
        Mapa de calor Grad-CAM
      </Typography>

      <Typography color="text.secondary" sx={{ mb: 2 }} />

      {/* Contenedor: manda la imagen más pequeña */}
      <Box
        sx={{
          position: 'relative',
          display: 'inline-block',
          mx: 'auto',
          overflow: 'hidden', // 🔒 recorte
        }}
      >
        {/* Imagen base */}
        <Box
          component="img"
          src={image}
          alt="Original"
          sx={{
            display: 'block',
            maxWidth: 500,
            width: 'auto',
            height: 'auto',
            borderRadius: 2,
          }}
        />

        {/* Heatmap: se recorta por ABAJO si sobra */}
        {showGradCam && (
          <Box
            component="img"
            src={heatmap}
            alt="Grad-CAM"
            sx={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover',        // 🔑 recorta
              objectPosition: 'top',     // 🔑 corta abajo
              opacity: 0.55,
              borderRadius: 2,
              pointerEvents: 'none',
            }}
          />
        )}

        {/* Toggle */}
        <Tooltip title={showGradCam ? 'Ocultar Grad-CAM' : 'Mostrar Grad-CAM'}>
          <IconButton
            onClick={() => setShowGradCam(prev => !prev)}
            sx={{
              position: 'absolute',
              top: 8,
              right: 8,
              bgcolor: 'rgba(0,0,0,0.6)',
              color: 'white',
              '&:hover': {
                bgcolor: 'rgba(0,0,0,0.8)',
              },
            }}
            size="small"
          >
            {showGradCam ? <VisibilityOffIcon /> : <VisibilityIcon />}
          </IconButton>
        </Tooltip>
      </Box>
    </>
  )
}
