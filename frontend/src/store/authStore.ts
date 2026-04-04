import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { Physician } from '@/types'

interface AuthState {
  physician: Physician | null
  login: (id: string, password: string) => Promise<void>
  logout: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      physician: null,

      login: async (id: string, password: string) => {
        // Mock phase: accept any non-empty credentials
        if (!id.trim() || !password.trim()) {
          throw new Error('Physician ID and password are required')
        }
        // Simulate network latency
        await new Promise(res => setTimeout(res, 500))
        set({ physician: { id: 'DR-001', name: 'Dr. D. Cohen', initials: 'DC' } })
      },

      logout: () => {
        set({ physician: null })
      },
    }),
    {
      name: 'oncoflow_auth',
      storage: createJSONStorage(() => sessionStorage),
    }
  )
)
