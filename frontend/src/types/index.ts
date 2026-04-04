export type PatientStatus = 'active' | 'review'

export interface Patient {
  id: string               // e.g. "P-1029"
  name: string
  dob: string              // ISO date "YYYY-MM-DD"
  diagnosis: string
  diagnosisLocation: string
  assignedPhysicianId: string
  status: PatientStatus
  scanCount: number
  lastScanDate: string     // ISO date
}

export interface Scan {
  id: string               // e.g. "SCN-0041"
  patientId: string
  studyLabel: string       // e.g. "MRI Study #3"
  date: string             // ISO date
  modality: string         // e.g. "MRI"
  sequence: string         // e.g. "T1W"
  plane: string            // e.g. "AXIAL"
  sliceCount: number
  resolution: string       // e.g. "1.2mm iso"
  volumeMm3: number
  maxDiameterMm: number
  isAnnotated: boolean
}

export interface Summary {
  patientId: string
  generatedAt: string      // ISO datetime
  model: string
  text: string
}

export interface MriUrl {
  url: string
  expiresAt: string        // ISO datetime
}

export interface Physician {
  id: string               // e.g. "DR-001"
  name: string
  initials: string
}
