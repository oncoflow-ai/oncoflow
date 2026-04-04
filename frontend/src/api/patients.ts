import type { Patient } from '@/types'
import { mockPatients } from '@/data/mockData'
import { delay } from './client'

export async function getPatients(): Promise<Patient[]> {
  await delay(400)
  return mockPatients
}

export async function getPatient(id: string): Promise<Patient> {
  await delay(300)
  const patient = mockPatients.find(p => p.id === id)
  if (!patient) throw new Error(`Patient ${id} not found`)
  return patient
}
