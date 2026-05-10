import { renderHook, act } from '@testing-library/react'
import { demoUsers, useAuthStore } from '@/store/authStore'

beforeEach(() => {
  useAuthStore.setState({ user: null, users: demoUsers, patientAssignments: {} })
})

describe('useAuthStore', () => {
  it('starts unauthenticated', () => {
    const { result } = renderHook(() => useAuthStore())
    expect(result.current.user).toBeNull()
  })

  it('login sets user by email', async () => {
    const { result } = renderHook(() => useAuthStore())
    await act(async () => {
      await result.current.login('dr.cohen@ichilov.gov.il', 'password')
    })
    expect(result.current.user?.name).toBe('Dr. D. Cohen')
    expect(result.current.user?.role).toBe('doctor')
  })

  it('login sets user by id and role for legacy tests', async () => {
    const { result } = renderHook(() => useAuthStore())
    await act(async () => {
      await result.current.login('DR-001', 'password', 'doctor')
    })
    expect(result.current.user?.name).toBe('Dr. D. Cohen')
    expect(result.current.user?.role).toBe('doctor')
  })

  it('login throws with empty credentials', async () => {
    const { result } = renderHook(() => useAuthStore())
    await expect(result.current.login('', '', 'doctor')).rejects.toThrow('required')
  })

  it('addUser creates a role-bearing user who can sign in', async () => {
    const { result } = renderHook(() => useAuthStore())
    await act(async () => {
      await result.current.addUser({
        name: 'Alex Reviewer',
        email: 'alex.reviewer@hospital.test',
        password: 'review123',
        role: 'radiologist',
      })
      await result.current.login('alex.reviewer@hospital.test', 'review123')
    })
    expect(result.current.user?.role).toBe('radiologist')
  })

  it('logout clears auth state', async () => {
    const { result } = renderHook(() => useAuthStore())
    await act(async () => {
      await result.current.login('radiology@oncoflow.local', 'password')
    })
    act(() => result.current.logout())
    expect(result.current.user).toBeNull()
  })
})
