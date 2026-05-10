import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Routes, Route } from 'react-router-dom'
import { demoUsers, useAuthStore } from '@/store/authStore'

function Protected() {
  const user = useAuthStore(s => s.user)
  if (user === null) return <div>Redirected to auth</div>
  return <div>Protected content</div>
}

beforeEach(() => {
  useAuthStore.setState({ user: null, users: demoUsers, patientAssignments: {} })
})

describe('route protection', () => {
  it('shows redirect message when not authenticated', () => {
    render(
      <MemoryRouter initialEntries={['/doctor']}>
        <Routes>
          <Route path="/doctor" element={<Protected />} />
        </Routes>
      </MemoryRouter>
    )
    expect(screen.getByText('Redirected to auth')).toBeInTheDocument()
  })

  it('shows content when authenticated', async () => {
    await useAuthStore.getState().login('DR-001', 'password', 'doctor')
    render(
      <MemoryRouter initialEntries={['/doctor']}>
        <Routes>
          <Route path="/doctor" element={<Protected />} />
        </Routes>
      </MemoryRouter>
    )
    expect(screen.getByText('Protected content')).toBeInTheDocument()
  })
})
