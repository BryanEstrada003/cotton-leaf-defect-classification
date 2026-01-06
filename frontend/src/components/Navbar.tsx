import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material'
import { useNavigate } from 'react-router-dom'

export default function Navbar() {
  const navigate = useNavigate()

  return (
    <AppBar position="static" elevation={0}>
      <Toolbar>
        <Typography
          variant="h6"
          sx={{ flexGrow: 1, cursor: 'pointer' }}
          onClick={() => navigate('/')}
        >
          CotVision
        </Typography>

        <Box>
          <Button color="inherit" onClick={() => navigate('/')}>
            Inicio
          </Button>
          <Button color="inherit" onClick={() => navigate('/analyze')}>
            Analizar
          </Button>
          <Button color="inherit" onClick={() => navigate('/about')}>
            Proyecto
          </Button>
          <Button color="inherit" onClick={() => navigate('/history')}>
            Historial
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  )
}
