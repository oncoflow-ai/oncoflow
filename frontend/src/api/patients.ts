import type { Patient } from '@/types'
import { mockPatients } from '@/data/mockData'
import { delay } from './client'

const EXTRA_KEY = 'oncoflow_mock_patients_extra'

function readExtras(): Patient[] {
  try {
    const raw = sessionStorage.getItem(EXTRA_KEY)
    if (!raw) return []
    return JSON.parse(raw) as Patient[]
  } catch {
    return []
  }
}

function writeExtras(patients: Patient[]) {
  sessionStorage.setItem(EXTRA_KEY, JSON.stringify(patients))
}

import { readExtraScans } from './scans'
import { mockScans } from '@/data/mockData'

function computePatientWithScans(p: Patient): Patient {
  const extraScans = readExtraScans()
  const baseScans = mockScans[p.id] ?? []
  const extras = extraScans[p.id] ?? []
  const allScans = [...baseScans, ...extras]
  if (allScans.length > 0) {
    allScans.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime())
    const latest = allScans[allScans.length - 1]
    return {
      ...p,
      scanCount: allScans.length,
      lastScanDate: latest.date,
    }
  }
  return p
}

function mergedPatients(): Patient[] {
  const seen = new Set<string>()
  const out: Patient[] = []
  for (const p of mockPatients) {
    seen.add(p.id)
    out.push(computePatientWithScans(p))
  }
  for (const p of readExtras()) {
    if (!seen.has(p.id)) {
      seen.add(p.id)
      out.push(computePatientWithScans(p))
    }
  }
  return out
}

export async function getPatients(): Promise<Patient[]> {
  await delay(400)
  return mergedPatients()
}

export async function getPatient(id: string): Promise<Patient> {
  await delay(300)
  const patient = mergedPatients().find(p => p.id === id)
  if (!patient) throw new Error(`Patient ${id} not found`)
  return patient
}

export async function createPatient(input: {
  name: string
  dob: string
  diagnosis: string
  diagnosisLocation: string
  assignedPhysicianId: string
}): Promise<Patient> {
  await delay(350)
  const id = `P-${Date.now().toString(36).toUpperCase().slice(-6)}`
  const patient: Patient = {
    id,
    name: input.name.trim(),
    dob: input.dob,
    diagnosis: input.diagnosis.trim(),
    diagnosisLocation: input.diagnosisLocation.trim(),
    assignedPhysicianId: input.assignedPhysicianId,
    status: 'active',
    scanCount: 0,
    lastScanDate: input.dob,
  }
  const extras = readExtras()
  extras.push(patient)
  writeExtras(extras)
  return patient
}
