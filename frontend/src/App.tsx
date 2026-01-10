import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'

import theme from './app/theme'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Layout from './components/Layout'

import About from './pages/About'
import Analyze from './pages/Analyze'
import Result from './pages/Result'
import Catalog from './pages/Catalog'
import History from './pages/History'

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Box sx={{ minHeight: '100vh', minWidth: '100vw', bgcolor: 'background.default' }}>
          <Navbar />

          <Routes>
            <Route element={<Layout />}>
              <Route path="/about" element={<About />} />
              <Route path="/" element={<Analyze />} />
              <Route path="/result" element={<Result />} />
              <Route path="/catalog" element={<Catalog />} />
              <Route path="/history" element={<History />} />
            </Route>
          </Routes>

          <Footer />
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  )
}
