import type { Scan } from '@/types'
import { mockScans } from '@/data/mockData'
import { delay } from './client'

const SCANS_EXTRA_KEY = 'oncoflow_mock_scans_extra'

export function readExtraScans(): Record<string, Scan[]> {
  try {
    const raw = sessionStorage.getItem(SCANS_EXTRA_KEY)
    if (!raw) return {}
    return JSON.parse(raw) as Record<string, Scan[]>
  } catch {
    return {}
  }
}

export function saveMockScan(scan: Scan) {
  const extra = readExtraScans()
  const list = extra[scan.patientId] ?? []
  const idx = list.findIndex(s => s.id === scan.id)
  if (idx >= 0) {
    list[idx] = scan
  } else {
    list.push(scan)
  }
  extra[scan.patientId] = list
  sessionStorage.setItem(SCANS_EXTRA_KEY, JSON.stringify(extra))
}

export async function getScans(patientId: string): Promise<Scan[]> {
  await delay(350)
  const base = mockScans[patientId] ?? []
  const extra = readExtraScans()[patientId] ?? []
  const seen = new Set<string>()
  const merged: Scan[] = []
  for (const s of [...base, ...extra]) {
    if (!seen.has(s.id)) {
      seen.add(s.id)
      merged.push(s)
    }
  }
  return merged.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
}

export async function getScan(scanId: string): Promise<Scan> {
  await delay(200)
  const extra = readExtraScans()
  const allScans = [...Object.values(mockScans).flat(), ...Object.values(extra).flat()]
  const scan = allScans.find(s => s.id === scanId)
  if (scan) return scan
  throw new Error(`Scan ${scanId} not found`)
}
