import { Card, CardContent, Typography, Button } from '@mui/material'

interface Props {
  review: any
  onEdit: () => void
}

export default function ReviewCard({ review, onEdit }: Props) {
  return (
    <Card>
      <CardContent>
        <Typography variant="h5">
          Revisión
        </Typography>

        <Typography variant="h6" color="primary">
          {review.reviewed_class}
        </Typography>

        {review.comments && (
          <Typography color="text.secondary">
            {review.comments}
          </Typography>
        )}

        <Button variant="outlined" onClick={onEdit} sx={{ mt: 2 }}>
          Editar revisión
        </Button>
      </CardContent>
    </Card>
  )
}
