import { renderHook, act } from '@testing-library/react'
import { useAuthStore } from '@/store/authStore'

// Reset store between tests
beforeEach(() => {
  useAuthStore.setState({ physician: null })
})

describe('useAuthStore', () => {
  it('starts unauthenticated', () => {
    const { result } = renderHook(() => useAuthStore())
    expect(result.current.physician).toBeNull()
  })

  it('login sets physician', async () => {
    const { result } = renderHook(() => useAuthStore())
    await act(async () => {
      await result.current.login('DR-001', 'password')
    })
    expect(result.current.physician?.name).toBe('Dr. D. Cohen')
  })

  it('login throws with empty credentials', async () => {
    const { result } = renderHook(() => useAuthStore())
    await expect(
      act(async () => result.current.login('', ''))
    ).rejects.toThrow('required')
  })

  it('logout clears auth state', async () => {
    const { result } = renderHook(() => useAuthStore())
    await act(async () => {
      await result.current.login('DR-001', 'password')
    })
    act(() => result.current.logout())
    expect(result.current.physician).toBeNull()
  })
})
