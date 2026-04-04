import { mockScans } from '@/data/mockData'

describe('ImagingHistory sort', () => {
  it('sorts scans newest first', () => {
    const scans = mockScans['P-1029']
    const sorted = [...scans].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    expect(sorted[0].date).toBe('2026-03-08')
    expect(sorted[sorted.length - 1].date).toBe('2025-09-14')
  })
})
