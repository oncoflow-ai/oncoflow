import { renderHook, act } from '@testing-library/react'
import { vi } from 'vitest'
import { apiClient } from '@/api/client'
import { demoUsers, useAuthStore } from '@/store/authStore'
import type { AuthenticatedUser } from '@/types'

beforeEach(() => {
  sessionStorage.clear()
  vi.mocked(apiClient.post).mockClear()
  useAuthStore.setState({ user: null, users: demoUsers, patientAssignments: {} })
})

describe('useAuthStore', () => {
  it('starts unauthenticated', () => {
    const { result } = renderHook(() => useAuthStore())
    expect(result.current.user).toBeNull()
  })

  it('login stores the backend-issued token and a sanitized backend user', async () => {
    vi.mocked(apiClient.post).mockResolvedValueOnce({
      data: {
        access_token: 'backend-issued-token',
        user: {
          id: 'DR-999',
          name: 'Dr. Backend',
          email: 'backend.doctor@hospital.test',
          role: 'doctor',
          password: 'must-not-reach-the-store',
        },
      },
    })
    const { result } = renderHook(() => useAuthStore())
    let authenticatedUser: AuthenticatedUser
    await act(async () => {
      authenticatedUser = await result.current.login('backend.doctor@hospital.test', 'password')
    })
    expect(apiClient.post).toHaveBeenCalledWith(
      '/api/v1/auth/login',
      expect.any(URLSearchParams),
      expect.objectContaining({ headers: { 'Content-Type': 'application/x-www-form-urlencoded' } })
    )
    expect(authenticatedUser).toEqual({
      id: 'DR-999',
      name: 'Dr. Backend',
      email: 'backend.doctor@hospital.test',
      role: 'doctor',
      initials: 'DB',
    })
    expect(result.current.user).toEqual(authenticatedUser)
    expect(sessionStorage.getItem('oncoflow_token')).toBe('backend-issued-token')
  })

  it('propagates a backend rejection without authenticating a demo user', async () => {
    const backendError = new Error('Backend rejected this sign-in')
    vi.mocked(apiClient.post).mockRejectedValueOnce(backendError)
    const { result } = renderHook(() => useAuthStore())

    await expect(
      result.current.login('dr.cohen@ichilov.gov.il', 'password', 'doctor')
    ).rejects.toBe(backendError)

    expect(result.current.user).toBeNull()
    expect(sessionStorage.getItem('oncoflow_token')).toBeNull()
  })

  it('rejects a malformed backend response without creating a session', async () => {
    vi.mocked(apiClient.post).mockResolvedValueOnce({
      data: {
        access_token: '',
        user: {
          id: 'DR-999',
          name: 'Dr. Backend',
          email: 'backend.doctor@hospital.test',
          role: 'doctor',
        },
      },
    })
    const { result } = renderHook(() => useAuthStore())

    await expect(
      result.current.login('backend.doctor@hospital.test', 'password')
    ).rejects.toThrow('invalid')

    expect(result.current.user).toBeNull()
    expect(sessionStorage.getItem('oncoflow_token')).toBeNull()
  })

  it('login uses a successful backend response for an ID and ignores the legacy role argument', async () => {
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

  it('addUser creates a role-bearing user without changing backend-only sign-in', async () => {
    const { result } = renderHook(() => useAuthStore())
    await act(async () => {
      await result.current.addUser({
        name: 'Alex Reviewer',
        email: 'alex.reviewer@hospital.test',
        password: 'review123',
        role: 'radiologist',
      })
    })
    await act(async () => {
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
