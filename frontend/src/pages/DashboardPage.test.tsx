import { beforeEach, describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { useAuthStore } from '@/store/authStore'
import DashboardPage from '@/pages/DashboardPage'

describe('DashboardPage redirect', () => {
  beforeEach(() => {
    useAuthStore.setState({ user: null })
  })

  it('sends authenticated doctors to /doctor', async () => {
    await useAuthStore.getState().login('DR-001', 'pw', 'doctor')
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/doctor" element={<div>Doctor home</div>} />
        </Routes>
      </MemoryRouter>
    )
    expect(await screen.findByText('Doctor home')).toBeInTheDocument()
  })

  it('sends authenticated radiologists to /radiologist', async () => {
    await useAuthStore.getState().login('RAD-001', 'pw', 'radiologist')
    render(
      <MemoryRouter initialEntries={['/dashboard']}>
        <Routes>
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/radiologist" element={<div>Radiologist home</div>} />
        </Routes>
      </MemoryRouter>
    )
    expect(await screen.findByText('Radiologist home')).toBeInTheDocument()
  })
})
