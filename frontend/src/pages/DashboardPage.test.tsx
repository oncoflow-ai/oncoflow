import { describe, it, expect } from 'vitest'
import { mockPatients } from '@/data/mockData'

describe('dashboard patient filter', () => {
  function filterPatients(patients: typeof mockPatients, query: string) {
    const q = query.trim().toLowerCase()
    if (!q) return patients
    return patients.filter(p =>
      p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q)
    )
  }

  it('returns all patients when query is empty', () => {
    expect(filterPatients(mockPatients, '').length).toBe(mockPatients.length)
  })

  it('filters by patient name case-insensitively', () => {
    const result = filterPatients(mockPatients, 'sarah')
    expect(result.length).toBe(1)
    expect(result[0].name).toBe('Sarah Jenkins')
  })

  it('filters by patient ID', () => {
    const result = filterPatients(mockPatients, 'P-1031')
    expect(result.length).toBe(1)
    expect(result[0].id).toBe('P-1031')
  })

  it('returns empty when no match', () => {
    expect(filterPatients(mockPatients, 'zzz').length).toBe(0)
  })
})
