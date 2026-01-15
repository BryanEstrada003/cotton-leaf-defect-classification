import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  Switch,
} from '@mui/material'
import { useNavigate } from 'react-router-dom'
import { useAppStore } from '../stores/appMode'

export default function Navbar() {
  const navigate = useNavigate()
  const { isTechnician, setIsTechnician } = useAppStore()

  const handleToggleMode = () => {
    setIsTechnician(!isTechnician)
  }

  return (
    <AppBar position="static" elevation={0}>
      <Toolbar>
        {isTechnician ? (
        <Typography
          variant="h6"
          sx={{ flexGrow: 1, cursor: 'pointer' }}
          onClick={() => navigate('/')}
        >
          Analizar
        </Typography>
        ) : (
          <Typography
          variant="h6"
          sx={{ flexGrow: 1, cursor: 'pointer' }}
          onClick={() => navigate('/')}
        >
          Diagnósticos
        </Typography>
        )}

        <Box display="flex" alignItems="center" gap={2}>
          {/* Botones existentes */}
          
          {isTechnician ? (<Button color="inherit" onClick={() => navigate('/history')}>
            Diagnósticos
          </Button> ) : null}
          <Button color="inherit" onClick={() => navigate('/catalog')}>
            Catálogo
          </Button>
          <Button color="inherit" onClick={() => navigate('/about')}>
            CotVision
          </Button>

          {/* SWITCH DE MODO */}
          <Box display="flex" alignItems="center" ml={2}>
            <Typography variant="body2">Técnico</Typography>
            <Switch
              checked={!isTechnician}
              onChange={handleToggleMode}
              color="default"
            />
            <Typography variant="body2">Productor</Typography>
          </Box>
        </Box>
      </Toolbar>
    </AppBar>
  )
}
