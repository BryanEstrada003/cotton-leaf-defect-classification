// components/LoadingOverlay.tsx
import { Backdrop, CircularProgress, Typography, Box } from '@mui/material'

type Props = {
  open: boolean
  text?: string
}

export default function LoadingOverlay({
  open,
  text = 'Procesando imagen…',
}: Props) {
  return (
    <Backdrop
      open={open}
      sx={{
        zIndex: (theme) => theme.zIndex.modal + 1,
        backdropFilter: 'blur(6px)',
        backgroundColor: 'rgba(0,0,0,0.4)',
      }}
    >
      <Box textAlign="center">
        <CircularProgress color="inherit" />
        <Typography variant="h6" sx={{ mt: 2 }}>
          {text}
        </Typography>
      </Box>
    </Backdrop>
  )
}
