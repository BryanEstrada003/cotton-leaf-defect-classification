import {
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  Box,
  Chip,
  Stack,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Container,
} from '@mui/material'
import { useNavigate } from 'react-router-dom'
import {
  ImageSearch,
  Psychology,
  History,
  VerifiedUser,
  PrecisionManufacturing,
  Devices,
  Agriculture,
  Engineering,
  Group,
  School,
} from '@mui/icons-material'

export default function About() {
  const navigate = useNavigate()

  const features = [
    {
      icon: <ImageSearch />,
      title: 'Análisis de Imágenes',
      items: [
        'Subida intuitiva de fotografías de hojas',
        'Preprocesamiento automático y validación',
        'Resultados detallados con nivel de confianza',
      ],
    },
    {
      icon: <Psychology />,
      title: 'Modelo KAN de IA',
      items: [
        'Arquitectura Redes de Kolmogorov-Arnold',
        'Alta precisión en reconocimiento visual',
        'Procesamiento en tiempo real',
      ],
    },
    {
      icon: <VerifiedUser />,
      title: 'Revisión por Profesionales',
      items: [
        'Validación experta por agrónomos',
        'Sistema de comentarios y correcciones',
      ],
    },
  ]

  const benefits = [
    {
      audience: 'Para Agricultores',
      icon: <Agriculture />,
      items: [
        'Diagnóstico en segundos desde cualquier dispositivo',
        'Interfaz diseñada para usuarios sin conocimientos técnicos',
        'Detección temprana para prevenir propagación',
      ],
    },
    {
      audience: 'Para Técnicos Agrícolas',
      icon: <Engineering />,
      items: [
        'Herramienta de apoyo complementaria al expertise profesional',
        'Gestión centralizada de múltiples cultivos',
      ],
    },
  ]

  const teamMembers = [
    'Melissa Ayllón Gutiérrez',
    'Michael Estrada Santana',
    'Juan Pablo Plúas Muñoz',
  ]

  return (
    <Container maxWidth="lg">
      {/* Header - Centrado */}
      <Box sx={{ 
        textAlign: 'center', 

        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <Typography 
          variant="h2" 
          component="h1" 
          gutterBottom 
          fontWeight="bold"
          sx={{ textAlign: 'center' }}
        >
          CotVision
        </Typography>
        <Chip
          label="Plataforma de Diagnóstico Inteligente"
          color="primary"
          sx={{ mb: 2}}
        />
        <Typography 
          variant="h5" 
          color="text.secondary" 
          paragraph
          sx={{ 
            maxWidth: '800px',
            textAlign: 'center',
            mx: 'auto'
          }}
        >
          Diagnóstico automático de enfermedades foliares en cultivos de algodón utilizando inteligencia artificial
        </Typography>
      </Box>

      {/* Funcionalidades Principales - Centrado */}
      <Box sx={{ textAlign: 'center', mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Funcionalidades Principales
        </Typography>
      </Box>
      
      <Grid container justifyContent="center" spacing={4} sx={{ mb: 4 }}>
        {features.map((feature, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card sx={{ 
              height: '100%', 
              borderRadius: 2,
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center'
            }}>
              <CardContent sx={{ width: '100%' }}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  mb: 0,
                  justifyContent: 'center'
                }}>
                  {feature.icon}
                  <Typography variant="h6" sx={{ ml: 2 }}>
                    {feature.title}
                  </Typography>
                </Box>
                <Box sx={{ 
                  display: 'flex',
                  justifyContent: 'center'
                }}>
                  <List dense sx={{ textAlign: 'left' }}>
                    {feature.items.map((item, idx) => (
                      <ListItem key={idx} sx={{ justifyContent: 'center' }}>
                        <ListItemIcon sx={{ minWidth: 36 }}>
                          •
                        </ListItemIcon>
                        <ListItemText 
                          primary={item} 
                          primaryTypographyProps={{ align: 'left' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Beneficios - Centrado */}
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h4" gutterBottom>
          Beneficios de la Plataforma
        </Typography>
      </Box>
      
      <Grid container justifyContent="center" spacing={4} sx={{ mb: 8 }}>
        {benefits.map((benefit, index) => (
          <Grid item xs={12} md={6} key={index}>
            <Card sx={{ 
              height: '100%', 
              borderRadius: 2,
              textAlign: 'center',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center'
            }}>
              <CardContent sx={{ width: '100%' }}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  mb: 2,
                  justifyContent: 'center'
                }}>
                  {benefit.icon}
                  <Typography variant="h6" sx={{ ml: 2 }}>
                    {benefit.audience}
                  </Typography>
                </Box>
                <Stack 
                  spacing={1} 
                  sx={{ 
                    alignItems: 'center',
                    width: '100%'
                  }}
                >
                  {benefit.items.map((item, idx) => (
                    <Chip
                      key={idx}
                      label={item}
                      variant="outlined"
                      sx={{ 
                        justifyContent: 'center',
                        width: '100%',
                        maxWidth: '400px'
                      }}
                    />
                  ))}
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Características Técnicas - Centrado */}
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'center',
        mb: 8 
      }}>
        <Box sx={{ 
          maxWidth: '900px',
          textAlign: 'center',
          p: 4,
          borderRadius: 2,
          bgcolor: 'background.paper',
          boxShadow: 1,
          width: '100%'
        }}>
          <Typography variant="h5" gutterBottom>
            Características Técnicas
          </Typography>
          <Grid container justifyContent="center" spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column',
                alignItems: 'center',
                mb: 2
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <Devices sx={{ mr: 1 }} />
                  <Typography variant="subtitle1">Frontend</Typography>
                </Box>
                <Stack 
                  direction="row" 
                  spacing={1} 
                  flexWrap="wrap"
                  justifyContent="center"
                >
                  <Chip label="Diseño responsivo" size="small" />
                  <Chip label="Interfaz intuitiva" size="small" />
                  <Chip label="Compatibilidad móvil" size="small" />
                </Stack>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ 
                display: 'flex', 
                flexDirection: 'column',
                alignItems: 'center',
                mb: 2
              }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                  <PrecisionManufacturing sx={{ mr: 1 }} />
                  <Typography variant="subtitle1">Integración</Typography>
                </Box>
                <Stack 
                  direction="row" 
                  spacing={1} 
                  flexWrap="wrap"
                  justifyContent="center"
                >
                  <Chip label="Backend para procesamiento" size="small" />
                  <Chip label="Almacenamiento seguro" size="small" />
                </Stack>
              </Box>
            </Grid>
          </Grid>
        </Box>
      </Box>

      {/* Equipo de Desarrollo - Centrado */}
      <Box sx={{ display: 'flex', justifyContent: 'center', mb: 8 }}>
        <Card sx={{ 
          borderRadius: 2,
          maxWidth: '900px',
          textAlign: 'center',
          width: '100%'
        }}>
          <CardContent>
            <Typography variant="h5" gutterBottom color="primary">
              Equipo de Desarrollo
            </Typography>
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center', 
              mb: 3,
              justifyContent: 'center'
            }}>
              <Group sx={{ mr: 2 }} />
              <Typography variant="h6">Integrantes del Grupo 7:</Typography>
            </Box>
            <Grid container justifyContent="center" spacing={2} sx={{ mb: 3 }}>
              {teamMembers.map((member, index) => (
                <Grid item xs={12} md={4} key={index}>
                  <Chip
                    label={member}
                    variant="outlined"
                    sx={{ 
                      width: '100%',
                      maxWidth: '250px',
                      justifyContent: 'center'
                    }}
                  />
                </Grid>
              ))}
            </Grid>
            <Divider sx={{ my: 2 }} />
            <Grid container justifyContent="center" spacing={2}>
              <Grid item xs={12} md={6}>
                <Box sx={{ 
                  display: 'flex', 
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <School sx={{ mr: 1 }} />
                  <Typography variant="body1" sx={{ textAlign: 'center' }}>
                    <strong>Docente:</strong> Enrique Pelaez
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="body1" sx={{ textAlign: 'center' }}>
                  <strong>Institución:</strong> Escuela Superior Politécnica del Litoral (ESPOL)
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="body1" sx={{ textAlign: 'center' }}>
                  <strong>Período:</strong> PAO II 2025
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Box>

      {/* Call to Action - Centrado */}
      <Box sx={{ 
        textAlign: 'center', 
        display: 'flex',
        justifyContent: 'center'
      }}>
        <Button
          variant="contained"
          size="large"
          onClick={() => navigate('/')}
          sx={{ px: 8, py: 1.5 }}
        >
          Comenzar Análisis
        </Button>
      </Box>
    </Container>
  )
}