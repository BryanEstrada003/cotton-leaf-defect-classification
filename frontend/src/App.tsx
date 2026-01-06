import { ThemeProvider } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'

import theme from './app/theme'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Layout from './components/Layout'

import Home from './pages/Home'
import Analyze from './pages/Analyze'
import Result from './pages/Result'
import About from './pages/About'

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Box sx={{ minHeight: '100vh', minWidth: '100vw', bgcolor: 'background.default' }}>
          <Navbar />

          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<Home />} />
              <Route path="/analyze" element={<Analyze />} />
              <Route path="/result" element={<Result />} />
              <Route path="/about" element={<About />} />
            </Route>
          </Routes>

          <Footer />
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  )
}
