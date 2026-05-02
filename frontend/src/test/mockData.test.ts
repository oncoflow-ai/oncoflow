import { mockPatients, mockScans, mockSummaries } from '@/data/mockData'

describe('mockPatients', () => {
  it('has 9 patients', () => {
    expect(mockPatients).toHaveLength(9)
  })
  it('all patients have required fields', () => {
    for (const p of mockPatients) {
      expect(p.id).toMatch(/^P-\d+$/)
      expect(p.name).toBeTruthy()
      expect(['active', 'review']).toContain(p.status)
    }
  })
})

describe('mockScans', () => {
  it('has scan data for all 9 patients', () => {
    expect(Object.keys(mockScans)).toHaveLength(9)
  })
  it('scan counts match patient scanCount', () => {
    for (const p of mockPatients) {
      expect(mockScans[p.id]).toHaveLength(p.scanCount)
    }
  })
})

describe('mockSummaries', () => {
  it('has summaries for all 9 patients', () => {
    expect(Object.keys(mockSummaries)).toHaveLength(9)
  })
  it('summary model field is set', () => {
    for (const s of Object.values(mockSummaries)) {
      expect(s.model).toBeTruthy()
    }
  })
})
