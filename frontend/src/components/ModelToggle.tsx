// components/ModelToggle.tsx
import { ToggleButton, ToggleButtonGroup } from '@mui/material'

interface Props {
  model: 'vgg16' | 'kan'
  onChange: (model: 'vgg16' | 'kan') => void
}

export default function ModelToggle({ model, onChange }: Props) {
  return (
    <ToggleButtonGroup
      value={model}
      exclusive
      onChange={(_, value) => {
        if (value) onChange(value)
      }}
      sx={{ mb: 3 }}
    >
      <ToggleButton value="vgg16">VGG16</ToggleButton>
      <ToggleButton value="kan">KAN</ToggleButton>
    </ToggleButtonGroup>
  )
}
