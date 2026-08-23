import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { apiClient } from '@/api/client'
import AuthPage from '@/pages/AuthPage'
import { demoUsers, useAuthStore } from '@/store/authStore'

beforeEach(() => {
  sessionStorage.clear()
  vi.mocked(apiClient.post).mockClear()
  useAuthStore.setState({ user: null, users: demoUsers, patientAssignments: {} })
})

function renderAuthPage() {
  return render(
    <MemoryRouter>
      <AuthPage />
      <LocationIndicator />
    </MemoryRouter>
  )
}

function LocationIndicator() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname}</output>
}

describe('AuthPage', () => {
  it('renders sign-in form', () => {
    renderAuthPage()
    expect(screen.getByText(/Clinical Sign In/i)).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/dr\.cohen/)).toBeInTheDocument()
  })

  it('toggles to register mode', async () => {
    renderAuthPage()
    await userEvent.click(screen.getByText(/Request access from admin/))
    expect(screen.getByText(/Request Access/i)).toBeInTheDocument()
  })

  it('shows error on empty credentials', async () => {
    renderAuthPage()
    await userEvent.click(screen.getByRole('button', { name: /Continue/i }))
    await screen.findByText(/required/i)
  })

  it('shows the backend sign-in error without authenticating or navigating', async () => {
    const user = userEvent.setup()
    vi.mocked(apiClient.post).mockRejectedValueOnce(new Error('Account is not authorized'))
    renderAuthPage()

    await user.type(screen.getByPlaceholderText(/dr\.cohen/), 'dr.cohen@ichilov.gov.il')
    await user.type(screen.getByPlaceholderText('••••••••'), 'password')
    await user.click(screen.getByRole('button', { name: /Continue/i }))

    expect(await screen.findByText('Account is not authorized')).toBeInTheDocument()
    expect(useAuthStore.getState().user).toBeNull()
    expect(sessionStorage.getItem('oncoflow_token')).toBeNull()
    expect(screen.getByTestId('location')).toHaveTextContent('/')
  })
})
