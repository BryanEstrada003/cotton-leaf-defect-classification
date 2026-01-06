import { Box, Typography } from '@mui/material'

interface Props {
  image: string
  heatmap: string
}

export default function GradCamOverlay({ image, heatmap }: Props) {
  return (
    <>
      <Typography variant="h6" sx={{ mt: 6, mb: 2 }}>
        Interpretabilidad del modelo (Grad-CAM)
      </Typography>

      <Typography color="text.secondary" sx={{ mb: 2 }}>
        Las zonas resaltadas en rojo indican las regiones de la hoja que
        influyeron más en la decisión del modelo.
      </Typography>

      <Box
        sx={{
          position: 'relative',
          width: '100%',
          maxWidth: 500,
          mx: 'auto',
        }}
      >
        {/* Imagen original */}
        <Box
          component="img"
          src={image}
          sx={{
            width: '100%',
            borderRadius: 2,
          }}
        />

        {/* Heatmap overlay */}
        <Box
          component="img"
          src={heatmap}
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            opacity: 0.55,
            borderRadius: 2,
          }}
        />
      </Box>
    </>
  )
}
