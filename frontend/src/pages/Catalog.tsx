import {
  Typography,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Divider,
} from '@mui/material'

import CV from '../assets/CV.jpg'
import BB from '../assets/BB.jpg'
import LR from '../assets/LR.jpg'

const diseases = [
  {
    name: 'Cotton Leaf Curl Virus (CLCuV)',
    image: CV,
    description:
      'Enfermedad viral altamente destructiva que afecta el crecimiento y rendimiento del algodón.',
    cause:
      'Es causada por un virus transmitido principalmente por la mosca blanca (Bemisia tabaci).',
    identification:
      'Hojas rizadas, engrosamiento de venas, crecimiento atrofiado y reducción de cápsulas.',
  },
  {
    name: 'Leaf Reddening',
    image: LR,
    description:
      'Trastorno que provoca el enrojecimiento progresivo de las hojas del algodón.',
    cause:
      'Deficiencia de nutrientes (especialmente potasio), estrés ambiental o infecciones.',
    identification:
      'Cambio de color verde a rojo o púrpura, principalmente en hojas maduras.',
  },
  {
    name: 'Bacterial Blight',
    image: BB,
    description:
      'Enfermedad bacteriana que afecta hojas, tallos y cápsulas del algodón.',
    cause:
      'Causada por la bacteria Xanthomonas citri pv. malvacearum en condiciones húmedas.',
    identification:
      'Manchas angulares oscuras, lesiones acuosas y daño en cápsulas.',
  },
]

export default function About() {
  return (
    <>
      <Typography variant="h4" gutterBottom>
        Enfermedades del algodón
      </Typography>

      <Typography paragraph color="text.secondary">
        Identificación visual de enfermedades foliares del algodón mediante
        inteligencia artificial.
      </Typography>

      <Grid
        container
        spacing={4}
        sx={{
          mt: 2,
          alignItems: 'stretch',
        }}
      >
        {diseases.map((disease) => (
          <Grid item xs={12} md={4} key={disease.name}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <CardMedia
                component="img"
                image={disease.image}
                alt={disease.name}
                sx={{
                  height: 220,
                  objectFit: 'cover',
                }}
              />

              <CardContent sx={{ flexGrow: 1 }}>
                <Typography variant="h6" gutterBottom>
                  {disease.name}
                </Typography>

                <Divider sx={{ mb: 1 }} />

                <Typography variant="subtitle2">Descripción</Typography>
                <Typography paragraph color="text.secondary">
                  {disease.description}
                </Typography>

                <Typography variant="subtitle2">Causa</Typography>
                <Typography paragraph color="text.secondary">
                  {disease.cause}
                </Typography>

                <Typography variant="subtitle2">Identificación</Typography>
                <Typography color="text.secondary">
                  {disease.identification}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </>
  )
}
