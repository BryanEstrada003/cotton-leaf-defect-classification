import { List, ListItem, ListItemText, Typography } from '@mui/material'

interface Props {
  recommendations: string[]
}

export default function RecommendationList({ recommendations }: Props) {
  return (
    <>
      <Typography sx={{ mt: 4 }} gutterBottom>
        Recomendaciones
      </Typography>

      <List>
        {recommendations.map((rec, idx) => (
          <ListItem key={idx}>
            <ListItemText primary={rec} />
          </ListItem>
        ))}
      </List>
    </>
  )
}
