import { getPatients, getPatient } from '@/api/patients'
import { getScans } from '@/api/scans'
import { getSummary } from '@/api/reports'

describe('getPatients()', () => {
  it('returns 8 patients', async () => {
    const patients = await getPatients()
    expect(patients).toHaveLength(8)
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
