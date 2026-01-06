import { Box, LinearProgress, Typography } from '@mui/material'

interface Props {
  confidence: number
}

export default function ConfidenceBar({ confidence }: Props) {
  const value = Math.round(confidence * 100)

  return (
    <Box sx={{ mt: 4 }}>
      <Typography gutterBottom>
        Confianza del modelo: {value}%
      </Typography>
      <LinearProgress
        variant="determinate"
        value={value}
        sx={{ height: 10, borderRadius: 5 }}
      />
    </Box>
  )
}
