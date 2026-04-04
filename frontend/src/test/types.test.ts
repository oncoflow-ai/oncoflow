import type { Patient, Scan, PatientStatus } from '@/types'

describe('type definitions compile', () => {
  it('Patient shape is correct', () => {
    const p: Patient = {
      id: 'P-1029',
      name: 'Test',
      dob: '1994-07-22',
      diagnosis: 'Osteosarcoma',
      diagnosisLocation: 'Distal Left Femur',
      assignedPhysicianId: 'DR-001',
      status: 'active',
      scanCount: 3,
      lastScanDate: '2026-03-08',
    }
    expect(p.id).toBe('P-1029')
  })

  it('PatientStatus is a union', () => {
    const s: PatientStatus = 'review'
    expect(['active', 'review']).toContain(s)
  })

  it('Scan shape is correct', () => {
    const s: Scan = {
      id: 'SCN-0041',
      patientId: 'P-1029',
      studyLabel: 'MRI Study #1',
      date: '2026-01-10',
      modality: 'MRI',
      sequence: 'T1W',
      plane: 'AXIAL',
      sliceCount: 128,
      resolution: '1.2mm iso',
      volumeMm3: 12480,
      maxDiameterMm: 28.4,
      isAnnotated: true,
    }
    expect(s.volumeMm3).toBe(12480)
  })
})
