import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import AuthPage from '@/pages/AuthPage'
import { useAuthStore } from '@/store/authStore'

beforeEach(() => {
  useAuthStore.setState({ physician: null })
})

function renderAuthPage() {
  return render(
    <MemoryRouter>
      <AuthPage />
    </MemoryRouter>
  )
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
    await userEvent.click(screen.getByText(/Access Patient Records/))
    await screen.findByText(/required/i)
  })
})
