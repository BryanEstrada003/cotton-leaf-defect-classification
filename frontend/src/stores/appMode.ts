import { create } from 'zustand'

interface AppModeState {
  isTechnician: boolean
  setIsTechnician: (value: boolean) => void
}

export const useAppStore = create<AppModeState>((set) => ({
  isTechnician: true, // true = Modo TÉCNICO | false = Modo PRODUCTOR
  setIsTechnician: (value) => set({ isTechnician: value }),
}))
