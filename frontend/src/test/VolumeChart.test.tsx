import { mockScans } from '@/data/mockData'

describe('VolumeChart data transform', () => {
  it('maps scans to chart data correctly', () => {
    const scans = mockScans['P-1029']
    const data = scans.map(s => ({ date: s.date, volume: s.volumeMm3 }))
    expect(data).toHaveLength(3)
    expect(data[0].volume).toBe(18400)
  })
})
