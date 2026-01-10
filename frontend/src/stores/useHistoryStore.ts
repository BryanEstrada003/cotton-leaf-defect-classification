import { create } from 'zustand'

export type Diagnosis = {
  id: string
  image: string
  result: any
  createdAt: string
  review?: {
    reviewed_class?: string
    comments?: string
    reviewedAt: string
    status: 'pending' | 'approved' | 'corrected'
  }
}

type HistoryStore = {
  history: Diagnosis[]
  addDiagnosis: (d: Diagnosis) => void
  updateDiagnosis: (id: string, data: Partial<Diagnosis>) => void
}

export const useHistoryStore = create<HistoryStore>((set) => ({
  history: [],
  addDiagnosis: (d) =>
    set((state) => ({ history: [d, ...state.history] })),

  updateDiagnosis: (id, data) =>
    set((state) => ({
      history: state.history.map((h) =>
        h.id === id ? { ...h, ...data } : h
      ),
    })),
}))
