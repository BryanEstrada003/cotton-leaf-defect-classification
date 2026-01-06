import { Box, CircularProgress, Typography } from '@mui/material'

export default function Loader() {
  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        mt: 8,
      }}
    >
      <CircularProgress size={60} />
      <Typography sx={{ mt: 2 }}>
        Analizando imagen con IA...
      </Typography>
    </Box>
  )
}
