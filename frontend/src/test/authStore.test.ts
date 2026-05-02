import { renderHook, act } from '@testing-library/react'
import { useAuthStore } from '@/store/authStore'

beforeEach(() => {
  useAuthStore.setState({ user: null })
})

describe('useAuthStore', () => {
  it('starts unauthenticated', () => {
    const { result } = renderHook(() => useAuthStore())
    expect(result.current.user).toBeNull()
  })

  it('login sets user', async () => {
    const { result } = renderHook(() => useAuthStore())
    await act(async () => {
      await result.current.login('DR-001', 'password', 'doctor')
    })
    expect(result.current.user?.name).toBe('Dr. D. Cohen')
    expect(result.current.user?.role).toBe('doctor')
  })

  it('login throws with empty credentials', async () => {
    const { result } = renderHook(() => useAuthStore())
    await expect(
      act(async () => result.current.login('', '', 'doctor'))
    ).rejects.toThrow('required')
  })

  it('logout clears auth state', async () => {
    const { result } = renderHook(() => useAuthStore())
    await act(async () => {
      await result.current.login('DR-001', 'password', 'radiologist')
    })
    act(() => result.current.logout())
    expect(result.current.user).toBeNull()
  })
})
