// components/RecommendationList.tsx
import { List, ListItem, ListItemText, Typography } from "@mui/material"

interface Props {
  items?: string[] // 👈 opcional
}

export default function RecommendationList({ items = [] }: Props) {
  if (items.length === 0) {
    return (
      <Typography color="text.secondary">
        No hay recomendaciones disponibles.
      </Typography>
    )
  }

  return (
    <List>
      {items.map((item, index) => (
        <ListItem key={index}>
          <ListItemText primary={item} />
        </ListItem>
      ))}
    </List>
  )
}
