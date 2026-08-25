import type { ClinicalReportEntry, Summary } from '@/types'
import { mockSummaries } from '@/data/mockData'
import { apiClient, delay } from './client'

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
  try {
    const res = await apiClient.get(`/api/v1/agents/summaries/${patientId}`)
    if (res.data && res.data.length > 0) {
      const latest = res.data[0]
      const text = `${latest.findings}\n\n**Impression:** ${latest.impression}\n\n**Interval Comparison:** ${latest.comparison}`
      return {
        patientId,
        generatedAt: latest.created_at,
        model: latest.model_name || 'oncoflow-multiagent-v1',
        text,
        findings: latest.findings,
        impression: latest.impression,
        comparison: latest.comparison,
        recommendations: latest.recommendations || [],
        ragSources: [],
      }
    }
  } catch {
    // Graceful fallback to mock data
  }

  await delay(400)
  const summary = mockSummaries[patientId]
  if (!summary) {
    return {
      patientId,
      generatedAt: new Date().toISOString(),
      model: 'oncoflow-multiagent-v1',
      text: 'AI Multi-agent coordination initialized. Awaiting MRI scan ingestion or clinical document index.',
      recommendations: ['Perform baseline brain MRI.', 'Upload patient clinical notes for RAG indexing.'],
    }
  }
  return summary
}

export async function listReports(patientId: string): Promise<ClinicalReportEntry[]> {
  const stored = readReports()[patientId] ?? []
  try {
    const res = await apiClient.get(`/api/v1/agents/summaries/${patientId}`)
    if (res.data && res.data.length > 0) {
      const clinicalReports = res.data.map((s: { summary_id: string; title: string; created_at: string; findings: string }) => ({
        id: s.summary_id,
        patientId,
        title: s.title || `Clinical Summary · ${s.created_at.slice(0, 10)}`,
        generatedAt: s.created_at,
        summarySnippet: s.findings.slice(0, 140).replace(/\s+/g, ' ').trim() + '…',
      }))
      return [...clinicalReports, ...stored]
        .filter((report, index, reports) => reports.findIndex(candidate => candidate.id === report.id) === index)
        .sort((a, b) => new Date(b.generatedAt).getTime() - new Date(a.generatedAt).getTime())
    }
  } catch {
    // Fallback to session storage
  }

  await delay(250)
  return [...stored].sort(
    (a, b) => new Date(b.generatedAt).getTime() - new Date(a.generatedAt).getTime()
  )
}

export async function generateReport(patientId: string): Promise<ClinicalReportEntry> {
  try {
    const res = await apiClient.post('/api/v1/agents/orchestrate-summary', {
      patient_id: patientId,
      custom_query: 'prior baseline tumor volume diameter response summary',
      persist: true,
    })
    if (res.data && res.data.summary) {
      const summary = res.data.summary
      const entry: ClinicalReportEntry = {
        id: res.data.orchestration_id,
        patientId,
        title: summary.title,
        generatedAt: res.data.completed_at,
        summarySnippet: summary.findings.slice(0, 140).replace(/\s+/g, ' ').trim() + '…',
      }
      const map = readReports()
      const prev = map[patientId] ?? []
      map[patientId] = [entry, ...prev]
      writeReports(map)
      return entry
    }
  } catch {
    // Fallback to simulated generation
  }

  await delay(500)
  const summary = mockSummaries[patientId]
  const snippet = summary
    ? summary.text.slice(0, 140).replace(/\s+/g, ' ').trim() + '…'
    : 'Generated multi-agent longitudinal assessment.'

  const entry: ClinicalReportEntry = {
    id: `RPT-${Date.now().toString(36)}`,
    patientId,
    title: `Multi-Agent Clinical Summary · ${new Date().toISOString().slice(0, 10)}`,
    generatedAt: new Date().toISOString(),
    summarySnippet: snippet,
  }

  const map = readReports()
  const prev = map[patientId] ?? []
  map[patientId] = [entry, ...prev]
  writeReports(map)
  return entry
}

import { saveMockScan } from './scans'

/**
 * MRI analyses are saved locally until the clinical-report backend owns this
 * relationship. The stable study-based id makes completion retries idempotent.
 */
export function saveMriAnalysisReport(patientId: string, studyId: string): ClinicalReportEntry {
  const entry: ClinicalReportEntry = {
    id: `MRI-${studyId}`,
    patientId,
    studyId,
    kind: 'mri-analysis',
    title: 'MRI segmentation analysis',
    generatedAt: new Date().toISOString(),
    summarySnippet: 'MRI segmentation is complete and ready for clinical review.',
  }

  const map = readReports()
  const reports = map[patientId] ?? []
  const existingIndex = reports.findIndex(report => report.studyId === studyId)
  map[patientId] = existingIndex === -1
    ? [entry, ...reports]
    : reports.map((report, index) => index === existingIndex ? { ...report, ...entry, generatedAt: report.generatedAt } : report)
  writeReports(map)

  // Register newly analyzed MRI scan so it appears immediately in the Scans & Viewer and patient roster
  const scanDate = new Date().toISOString().slice(0, 10)
  saveMockScan({
    id: `SCN-${studyId.slice(0, 8)}`,
    patientId,
    studyLabel: 'MRI Study (AI Analyzed)',
    date: scanDate,
    modality: 'MRI',
    sequence: 'T1c',
    plane: 'AXIAL',
    sliceCount: 160,
    resolution: '1.0mm iso',
    volumeMm3: 14815,
    maxDiameterMm: 64.8,
    isAnnotated: true,
  })

  return entry
}
