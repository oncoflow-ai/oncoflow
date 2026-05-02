import type { ClinicalReportEntry, Summary } from '@/types'
import { mockSummaries } from '@/data/mockData'
import { delay } from './client'

const REPORTS_KEY = 'oncoflow_mock_reports'

function readReports(): Record<string, ClinicalReportEntry[]> {
  try {
    const raw = sessionStorage.getItem(REPORTS_KEY)
    if (!raw) return {}
    return JSON.parse(raw) as Record<string, ClinicalReportEntry[]>
  } catch {
    return {}
  }
}

function writeReports(map: Record<string, ClinicalReportEntry[]>) {
  sessionStorage.setItem(REPORTS_KEY, JSON.stringify(map))
}

export async function getSummary(patientId: string): Promise<Summary> {
  await delay(600)
  const summary = mockSummaries[patientId]
  if (!summary) throw new Error(`Summary for patient ${patientId} not found`)
  return summary
}

export async function listReports(patientId: string): Promise<ClinicalReportEntry[]> {
  await delay(250)
  const stored = readReports()[patientId] ?? []
  return [...stored].sort(
    (a, b) => new Date(b.generatedAt).getTime() - new Date(a.generatedAt).getTime()
  )
}

export async function generateReport(patientId: string): Promise<ClinicalReportEntry> {
  await delay(500)
  const summary = mockSummaries[patientId]
  const snippet = summary
    ? summary.text.slice(0, 140).replace(/\s+/g, ' ').trim() + '…'
    : 'Generated longitudinal assessment (mock).'

  const entry: ClinicalReportEntry = {
    id: `RPT-${Date.now().toString(36)}`,
    patientId,
    title: `Clinical summary · ${new Date().toISOString().slice(0, 10)}`,
    generatedAt: new Date().toISOString(),
    summarySnippet: snippet,
  }

  const map = readReports()
  const prev = map[patientId] ?? []
  map[patientId] = [entry, ...prev]
  writeReports(map)
  return entry
}
