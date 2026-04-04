import { mockPatients, mockScans, mockSummaries } from '@/data/mockData'

describe('mockPatients', () => {
  it('has 8 patients', () => {
    expect(mockPatients).toHaveLength(8)
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
  it('has scan data for all 8 patients', () => {
    expect(Object.keys(mockScans)).toHaveLength(8)
  })
  it('scan counts match patient scanCount', () => {
    for (const p of mockPatients) {
      expect(mockScans[p.id]).toHaveLength(p.scanCount)
    }
  })
})

describe('mockSummaries', () => {
  it('has summaries for all 8 patients', () => {
    expect(Object.keys(mockSummaries)).toHaveLength(8)
  })
  it('summary model field is set', () => {
    for (const s of Object.values(mockSummaries)) {
      expect(s.model).toBeTruthy()
    }
  })
})
