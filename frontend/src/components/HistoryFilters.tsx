import {
  Box,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from '@mui/material'

interface Props {
  classFilter: string
  modelFilter: string
  order: 'asc' | 'desc'
  onClassChange: (v: string) => void
  onModelChange: (v: string) => void
  onOrderChange: (v: 'asc' | 'desc') => void
  classes: string[]
}

export default function HistoryFilters({
  classFilter,
  modelFilter,
  order,
  onClassChange,
  onModelChange,
  onOrderChange,
  classes,
}: Props) {
  return (
    <Box sx={{ display: 'flex', gap: 2, mb: 3, flexWrap: 'wrap' }}>
      <FormControl sx={{ minWidth: 200 }}>
        <InputLabel>Clase</InputLabel>
        <Select
          value={classFilter}
          label="Clase"
          onChange={(e) => onClassChange(e.target.value)}
        >
          <MenuItem value="all">Todas</MenuItem>
          {classes.map((c) => (
            <MenuItem key={c} value={c}>
              {c}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <FormControl sx={{ minWidth: 160 }}>
        <InputLabel>Modelo</InputLabel>
        <Select
          value={modelFilter}
          label="Modelo"
          onChange={(e) => onModelChange(e.target.value)}
        >
          <MenuItem value="all">Todos</MenuItem>
          <MenuItem value="VGG16">VGG16</MenuItem>
          <MenuItem value="KAN">KAN</MenuItem>
        </Select>
      </FormControl>

      <FormControl sx={{ minWidth: 200 }}>
        <InputLabel>Orden</InputLabel>
        <Select
          value={order}
          label="Orden"
          onChange={(e) => onOrderChange(e.target.value as 'asc' | 'desc')}
        >
          <MenuItem value="desc">Más recientes</MenuItem>
          <MenuItem value="asc">Más antiguos</MenuItem>
        </Select>
      </FormControl>
    </Box>
  )
}
