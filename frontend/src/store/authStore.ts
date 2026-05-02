import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { AuthenticatedUser, UserRole } from '@/types'

interface AuthState {
  user: AuthenticatedUser | null
  login: (id: string, password: string, role: UserRole) => Promise<void>
  logout: () => void
}

function profileForRole(trimmedId: string, role: UserRole): AuthenticatedUser {
  if (role === 'doctor') {
    const id = trimmedId || 'DR-001'
    return {
      id,
      name: 'Dr. D. Cohen',
      initials: 'DC',
      role: 'doctor',
    }
  }
  if (role === 'radiologist') {
    const id = trimmedId || 'RAD-001'
    return {
      id,
      name: 'Alex Rahman',
      initials: 'AR',
      role: 'radiologist',
    }
  }
  const patientRecordId = trimmedId || 'P-1029'
  return {
    id: patientRecordId,
    name: 'Patient Portal',
    initials: 'ME',
    role: 'patient',
    patientRecordId,
  }
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,

      login: async (id: string, password: string, role: UserRole) => {
        const trimmed = id.trim()
        if (!trimmed || !password.trim()) {
          throw new Error('User ID and password are required')
        }
        if (role === 'patient' && !/^P-\d+/i.test(trimmed)) {
          throw new Error('Patient sign-in expects a patient ID such as P-1029')
        }
        await new Promise(res => setTimeout(res, 500))
        set({ user: profileForRole(trimmed, role) })
      },

      logout: () => {
        set({ user: null })
      },
    }),
    {
      name: 'oncoflow_auth_v2',
      storage: createJSONStorage(() => sessionStorage),
    }
  )
)
