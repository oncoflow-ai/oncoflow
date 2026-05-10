import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { AppUser, AuthenticatedUser, UserRole } from '@/types'

export const demoUsers: AppUser[] = [
  {
    id: 'ADM-001',
    name: 'Maya Administrator',
    initials: 'MA',
    email: 'admin@oncoflow.local',
    password: 'admin123',
    role: 'admin',
  },
  {
    id: 'DR-001',
    name: 'Dr. D. Cohen',
    initials: 'DC',
    email: 'dr.cohen@ichilov.gov.il',
    password: 'password',
    role: 'doctor',
  },
  {
    id: 'RAD-001',
    name: 'Alex Rahman',
    initials: 'AR',
    email: 'radiology@oncoflow.local',
    password: 'password',
    role: 'radiologist',
  },
  {
    id: 'CLN-001',
    name: 'Noa Clinical',
    initials: 'NC',
    email: 'clinician@oncoflow.local',
    password: 'password',
    role: 'clinician',
  },
  {
    id: 'PAT-1029',
    name: 'Sarah Jenkins',
    initials: 'SJ',
    email: 'sarah.jenkins@example.test',
    password: 'patient123',
    role: 'patient',
    patientRecordId: 'P-1029',
  },
]

interface AuthState {
  user: AuthenticatedUser | null
  users: AppUser[]
  patientAssignments: Record<string, string>
  login: (idOrEmail: string, password: string, role?: UserRole) => Promise<AuthenticatedUser>
  addUser: (user: Omit<AppUser, 'id' | 'initials'> & { id?: string }) => Promise<AppUser>
  assignPatient: (patientId: string, userId: string) => void
  logout: () => void
}

function sanitizeUser(user: AppUser): AuthenticatedUser {
  const { password: _password, ...authenticatedUser } = user
  return authenticatedUser
}

function initialsFromName(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map(part => part[0]?.toUpperCase())
    .join('') || 'U'
}

function mergeUsers(users: AppUser[] = []): AppUser[] {
  const byEmail = new Map<string, AppUser>()
  demoUsers.forEach(user => byEmail.set(user.email.toLowerCase(), user))
  users.forEach(user => byEmail.set(user.email.toLowerCase(), user))
  return Array.from(byEmail.values())
}

function findUser(users: AppUser[], idOrEmail: string, role?: UserRole): AppUser | undefined {
  const normalized = idOrEmail.trim().toLowerCase()
  return users.find(user => {
    const matchesLogin = user.email.toLowerCase() === normalized || user.id.toLowerCase() === normalized
    return matchesLogin && (!role || user.role === role)
  })
}

function makeUserId(role: UserRole, patientRecordId?: string): string {
  const rolePrefix: Record<UserRole, string> = {
    admin: 'ADM',
    doctor: 'DR',
    radiologist: 'RAD',
    clinician: 'CLN',
    patient: 'PAT',
  }
  return role === 'patient' && patientRecordId
    ? `${rolePrefix[role]}-${patientRecordId.replace('P-', '')}`
    : `${rolePrefix[role]}-${Date.now().toString().slice(-4)}`
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      users: demoUsers,
      patientAssignments: {},

      login: async (idOrEmail: string, password: string, role?: UserRole) => {
        if (!idOrEmail.trim() || !password.trim()) {
          throw new Error('User ID/email and password are required')
        }
        await new Promise(res => setTimeout(res, 500))

        const user = findUser(useAuthStore.getState().users, idOrEmail, role)
        if (!user || (!role && user.password !== password)) {
          throw new Error('Invalid user credentials')
        }

        const authenticatedUser = sanitizeUser(user)
        set({ user: authenticatedUser })
        return authenticatedUser
      },

      addUser: async ({ id, name, email, password, role, patientRecordId }) => {
        if (!name.trim() || !email.trim() || !password.trim()) {
          throw new Error('Name, email, and password are required')
        }
        if (role === 'patient' && !patientRecordId) {
          throw new Error('Patient accounts must be linked to a patient record')
        }

        const normalizedEmail = email.trim().toLowerCase()
        const existing = useAuthStore
          .getState()
          .users
          .some(user => user.email.toLowerCase() === normalizedEmail)

        if (existing) {
          throw new Error('A user with this email already exists')
        }

        const user: AppUser = {
          id: id ?? makeUserId(role, patientRecordId),
          name: name.trim(),
          initials: initialsFromName(name),
          email: normalizedEmail,
          password,
          role,
          patientRecordId: role === 'patient' ? patientRecordId : undefined,
        }

        set(state => ({ users: [...state.users, user] }))
        return user
      },

      assignPatient: (patientId: string, userId: string) => {
        set(state => ({
          patientAssignments: {
            ...(state.patientAssignments ?? {}),
            [patientId]: userId,
          },
        }))
      },

      logout: () => {
        set({ user: null })
      },
    }),
    {
      name: 'oncoflow_auth_v2',
      storage: createJSONStorage(() => sessionStorage),
      merge: (persisted, current) => {
        const state = persisted as Partial<AuthState> | undefined
        return {
          ...current,
          ...state,
          user: state?.user ?? null,
          users: mergeUsers(state?.users),
          patientAssignments: state?.patientAssignments ?? {},
        }
      },
    }
  )
)
