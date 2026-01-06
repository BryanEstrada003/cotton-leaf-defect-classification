import { createTheme } from '@mui/material/styles'

const theme = createTheme({
  palette: {
    primary: {
      main: '#2e7d32', // verde agrícola
    },
    secondary: {
      main: '#81c784',
    },
    background: {
      default: '#f4f6f5',
    },
  },
  shape: {
    borderRadius: 12,
  },
  typography: {
    fontFamily: 'Roboto, sans-serif',
  },
})

export default theme
