import type { Summary } from '@/types'
import { mockSummaries } from '@/data/mockData'
import { delay } from './client'

export async function getSummary(patientId: string): Promise<Summary> {
  await delay(600)
  const summary = mockSummaries[patientId]
  if (!summary) throw new Error(`Summary for patient ${patientId} not found`)
  return summary
}
