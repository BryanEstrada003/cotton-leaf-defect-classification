import { ToggleButton, ToggleButtonGroup } from "@mui/material"

interface Props {
  value: "vgg16" | "kan"
  onChange: (value: "vgg16" | "kan") => void
}

export default function ModelToggle({ value, onChange }: Props) {
  return (
    <ToggleButtonGroup
      exclusive
      value={value}
      onChange={(_, newValue) => {
        if (newValue !== null) {
          onChange(newValue)
        }
      }}
    >
      <ToggleButton value="vgg16">VGG16</ToggleButton>
      <ToggleButton value="kan">KAN</ToggleButton>
    </ToggleButtonGroup>
  )
}
