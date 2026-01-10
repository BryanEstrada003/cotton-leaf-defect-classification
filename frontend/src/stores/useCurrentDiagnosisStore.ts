// stores/useCurrentDiagnosisStore.ts
import { create } from 'zustand'
import type { Diagnosis } from './useHistoryStore'

type CurrentDiagnosisStore = {
  current: Diagnosis | null
  setCurrent: (d: Diagnosis) => void
  clearCurrent: () => void
}

export const useCurrentDiagnosisStore = create<CurrentDiagnosisStore>((set) => ({
  current: null,
  setCurrent: (d) => set({ current: d }),
  clearCurrent: () => set({ current: null }),
}))
