import { Grid, Typography } from '@mui/material'
import HistoryCard from '../components/HistoryCard'
import { useHistoryStore } from '../stores/useHistoryStore'

export default function History() {
  const items = useHistoryStore((s) => s.history)

  if (items.length === 0) {
    return <Typography variant="h5">No hay diagnósticos guardados.</Typography>
  }

  return (
    <>
      <Typography variant="h4" gutterBottom>
        Historial de diagnósticos
      </Typography>

      <Grid container spacing={3}>
        {items.map((item) => (
          <Grid item xs={12} sm={6} md={4} key={item.id}>
            <HistoryCard item={item} />
          </Grid>
        ))}
      </Grid>
    </>
  )
}
