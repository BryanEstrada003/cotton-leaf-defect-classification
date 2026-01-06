import {
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
} from '@mui/material'
import { useNavigate } from 'react-router-dom'

export default function Home() {
  const navigate = useNavigate()

  return (
    <>
      <Typography variant="h3" align="center" gutterBottom>
        CotVision
      </Typography>

      <Typography
        variant="h6"
        align="center"
        color="text.secondary"
        gutterBottom
      >
        Diagnóstico inteligente de enfermedades foliares en algodón
      </Typography>

      <Typography align="center" sx={{ mt: 2 }}>
        Plataforma basada en visión por computador e inteligencia artificial
        para apoyar la toma de decisiones agrícolas.
      </Typography>

      <Grid container spacing={4} sx={{ mt: 6 }}>
        {[
          {
            title: 'Detección Automática',
            desc: 'Identificación temprana de enfermedades en hojas de algodón.',
          },
          {
            title: 'Modelos de IA',
            desc: 'Comparación entre arquitecturas KAN y CNN tradicionales.',
          },
          {
            title: 'Interpretabilidad',
            desc: 'Visualización Grad-CAM para explicar decisiones del modelo.',
          },
        ].map((item) => (
          <Grid item xs={12} md={4} key={item.title}>
            <Card sx={{ height: '100%' }}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {item.title}
                </Typography>
                <Typography color="text.secondary">
                  {item.desc}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Grid container justifyContent="center" sx={{ mt: 6 }}>
        <Button
          variant="contained"
          size="large"
          onClick={() => navigate('/analyze')}
        >
          Analizar hoja
        </Button>
      </Grid>
    </>
  )
}
