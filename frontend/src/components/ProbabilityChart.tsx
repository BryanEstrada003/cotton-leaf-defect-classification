import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { Box, Typography } from '@mui/material'

interface Props {
  probabilities: Record<string, number>
}

export default function ProbabilityChart({ probabilities }: Props) {
  const data = Object.entries(probabilities).map(([key, value]) => ({
    name: key,
    value: Math.round(value * 100),
  }))

  return (
    <Box sx={{ mt: 4, height: 300 }}>
      <Typography gutterBottom>
        Probabilidades por clase (%)
      </Typography>

      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" fill="#2e7d32" />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  )
}
