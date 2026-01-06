import { Container, Box } from '@mui/material'
import { Outlet } from 'react-router-dom'

export default function Layout() {
  return (
    <Container
      maxWidth="lg"
      sx={{
        py: 6,
      }}
    >
      <Box>
        <Outlet />
      </Box>
    </Container>
  )
}
