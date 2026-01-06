import type { PredictionResult } from '../types/prediction'

export interface HistoryItem {
  id: string
  image: string
  result: PredictionResult
  model_used: 'VGG16' | 'KAN'
  created_at: string
}

export function filterAndSortHistory(
  items: HistoryItem[],
  classFilter: string,
  modelFilter: string,
  order: 'asc' | 'desc'
): HistoryItem[] {
  let filtered = [...items]

  if (classFilter !== 'all') {
    filtered = filtered.filter(
      (item) => item.result.class === classFilter
    )
  }

  if (modelFilter !== 'all') {
    filtered = filtered.filter(
      (item) => item.model_used === modelFilter
    )
  }

  filtered.sort((a, b) => {
    const dateA = new Date(a.created_at).getTime()
    const dateB = new Date(b.created_at).getTime()
    return order === 'asc' ? dateA - dateB : dateB - dateA
  })

  return filtered
}
