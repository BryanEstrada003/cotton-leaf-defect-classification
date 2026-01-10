import { Box, Typography } from '@mui/material'

export default function Footer() {
  return (
    <Box
      sx={{
        textAlign: 'center',
        py: 3,
        mt: 8,
        color: 'text.secondary',
      }}
    >
      <Typography variant="body2">
        CotVision © 2025 · Grupo 7 · Curso de Inteligencia Artificial · ESPOL
      </Typography>
    </Box>
  )
}
