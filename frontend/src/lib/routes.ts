import type { UserRole } from '@/types'

export const ROLE_HOME: Record<UserRole, string> = {
  admin: '/admin/users',
  doctor: '/doctor',
  clinician: '/doctor',
  radiologist: '/radiologist',
  patient: '/patient',
}
