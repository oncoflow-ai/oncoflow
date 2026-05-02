import type { UserRole } from '@/types'

export const ROLE_HOME: Record<UserRole, string> = {
  doctor: '/doctor',
  radiologist: '/radiologist',
  patient: '/patient',
}
