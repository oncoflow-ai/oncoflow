import { getPatients, getPatient } from '@/api/patients'
import { getScans } from '@/api/scans'
import { getSummary, listReports, saveMriAnalysisReport } from '@/api/reports'

describe('getPatients()', () => {
  it('returns 9 patients', async () => {
    const patients = await getPatients()
    expect(patients).toHaveLength(9)
  })
})

describe('getPatient()', () => {
  it('returns correct patient by id', async () => {
    const p = await getPatient('P-1029')
    expect(p.name).toBe('Sarah Jenkins')
  })
  it('throws for unknown id', async () => {
    await expect(getPatient('P-9999')).rejects.toThrow('not found')
  })
})

describe('getScans()', () => {
  it('returns scans for a patient', async () => {
    const scans = await getScans('P-1029')
    expect(scans).toHaveLength(3)
  })
  it('returns empty array for unknown patient', async () => {
    const scans = await getScans('P-9999')
    expect(scans).toHaveLength(0)
  })
})

describe('getSummary()', () => {
  it('returns summary for a patient', async () => {
    const s = await getSummary('P-1029')
    expect(s.patientId).toBe('P-1029')
    expect(s.text.length).toBeGreaterThan(100)
  })
})

describe('MRI analysis reports', () => {
  it('saves a completed analysis so it appears in the patient report list and scans list', async () => {
    sessionStorage.removeItem('oncoflow_mock_reports')
    sessionStorage.removeItem('oncoflow_mock_scans_extra')
    saveMriAnalysisReport('P-1029', 'study-123')

    const reports = await listReports('P-1029')
    expect(reports).toEqual(expect.arrayContaining([
      expect.objectContaining({
        patientId: 'P-1029',
        studyId: 'study-123',
        kind: 'mri-analysis',
      }),
    ]))

    const scans = await getScans('P-1029')
    expect(scans.some(s => s.id === 'SCN-study-12')).toBe(true)
  })

  it('provides demo BraTS scans and summary for Demo Patient P01 (P-9001)', async () => {
    const demoPatient = await getPatient('P-9001')
    expect(demoPatient.name).toBe('Demo Patient P01')
    expect(demoPatient.scanCount).toBe(2)

    const scans = await getScans('P-9001')
    expect(scans).toHaveLength(2)
    expect(scans[0].studyLabel).toContain('Baseline')
    expect(scans[1].studyLabel).toContain('Follow-up')
  })
})
