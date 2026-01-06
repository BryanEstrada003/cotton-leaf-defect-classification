import { Grid, Typography } from '@mui/material'
import { useEffect, useState } from 'react'
import HistoryCard from '../components/HistoryCard'

export default function History() {
  const [items, setItems] = useState<any[]>([])

  useEffect(() => {
    const raw = localStorage.getItem('history')
    if (raw) setItems(JSON.parse(raw))
  }, [])

  if (items.length === 0) {
    return <Typography>No hay diagnósticos guardados.</Typography>
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
