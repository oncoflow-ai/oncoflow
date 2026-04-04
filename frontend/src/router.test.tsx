import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'

function Protected() {
  const physician = useAuthStore(s => s.physician)
  if (physician === null) return <div>Redirected to auth</div>
  return <div>Protected content</div>
}

beforeEach(() => {
  useAuthStore.setState({ physician: null })
})

describe('route protection', () => {
  it('shows redirect message when not authenticated', () => {
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route path="/dashboard" element={<Protected />} />
        </Routes>
      </MemoryRouter>
    )
    expect(screen.getByText('Redirected to auth')).toBeInTheDocument()
  })

  it('shows content when authenticated', async () => {
    await useAuthStore.getState().login('DR-001', 'pw')
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route path="/dashboard" element={<Protected />} />
        </Routes>
      </MemoryRouter>
    )
    expect(screen.getByText('Protected content')).toBeInTheDocument()
  })
})
