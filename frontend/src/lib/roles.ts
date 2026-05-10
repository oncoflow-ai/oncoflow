import type { UserRole } from '@/types'

export const roleLabels: Record<UserRole, string> = {
  admin: 'Administrator',
  doctor: 'Oncologist',
  radiologist: 'Radiologist',
  clinician: 'Clinician',
  patient: 'Patient',
}

export const roleHomePaths: Record<UserRole, string> = {
  admin: '/admin/users',
  doctor: '/doctor',
  radiologist: '/radiologist',
  clinician: '/doctor',
  patient: '/patient',
}

export function getRoleHomePath(role: UserRole): string {
  return roleHomePaths[role]
}

export function canAccessRolePath(role: UserRole, pathname: string): boolean {
  if (pathname.startsWith('/admin')) return role === 'admin'
  if (pathname.startsWith('/review') || pathname.startsWith('/radiologist')) return role === 'radiologist'
  if (pathname.startsWith('/portal') || pathname.startsWith('/patient')) return role === 'patient'
  if (pathname.startsWith('/doctor')) return role === 'doctor' || role === 'clinician'
  if (pathname.startsWith('/dashboard') || pathname.startsWith('/patients')) return role !== 'admin'
  return true
}
