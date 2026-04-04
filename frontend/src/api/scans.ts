import type { Scan } from '@/types'
import { mockScans } from '@/data/mockData'
import { delay } from './client'

export async function getScans(patientId: string): Promise<Scan[]> {
  await delay(350)
  return mockScans[patientId] ?? []
}

export async function getScan(scanId: string): Promise<Scan> {
  await delay(200)
  for (const scans of Object.values(mockScans)) {
    const scan = scans.find(s => s.id === scanId)
    if (scan) return scan
  }
  throw new Error(`Scan ${scanId} not found`)
}
