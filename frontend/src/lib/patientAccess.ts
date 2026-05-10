import type { Patient, Physician } from '@/types'

export function getPatientOwnerId(
  patient: Patient,
  patientAssignments: Record<string, string> = {}
): string {
  return patientAssignments[patient.id] ?? patient.assignedPhysicianId
}

export function canViewPatient(
  physician: Physician | null,
  patient: Patient,
  patientAssignments: Record<string, string> = {}
): boolean {
  if (!physician) return false
  if (physician.role === 'admin' || physician.role === 'radiologist') return true
  if (physician.role === 'patient') return physician.patientRecordId === patient.id
  return getPatientOwnerId(patient, patientAssignments) === physician.id
}

export function filterPatientsForUser(
  patients: Patient[],
  physician: Physician | null,
  patientAssignments: Record<string, string> = {}
): Patient[] {
  return patients.filter(patient => canViewPatient(physician, patient, patientAssignments))
}
