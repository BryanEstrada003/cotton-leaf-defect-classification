import { Typography, Grid, Card, CardContent } from '@mui/material'

export default function About() {
  return (
    <>
      <Typography variant="h4" gutterBottom>
        Sobre el proyecto
      </Typography>

      <Typography paragraph>
        CotVision es un sistema de apoyo al diagnóstico agrícola basado en
        inteligencia artificial, diseñado para identificar enfermedades en hojas
        de algodón a partir de imágenes.
      </Typography>

      <Grid container spacing={4} sx={{ mt: 2 }}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Tecnologías utilizadas
              </Typography>
              <Typography color="text.secondary">
                React · TypeScript · Material UI · FastAPI · PyTorch · TensorFlow
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Objetivo
              </Typography>
              <Typography color="text.secondary">
                Facilitar la detección temprana y explicable de enfermedades
                foliares para mejorar la productividad agrícola.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </>
  )
}
