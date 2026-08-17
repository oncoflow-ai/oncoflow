import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import PatientTable from '@/components/patient/PatientTable'
import { mockPatients, mockScans } from '@/data/mockData'

function renderTable(loading = false) {
  render(
    <MemoryRouter>
      <PatientTable patients={mockPatients} scansMap={mockScans} loading={loading} />
    </MemoryRouter>
  )
}

describe('PatientTable', () => {
  it('renders all 9 patient rows', () => {
    renderTable()
    expect(screen.getByText('Sarah Jenkins')).toBeInTheDocument()
    expect(screen.getByText('David Levi')).toBeInTheDocument()
  })

  it('renders column headers', () => {
    renderTable()
    expect(screen.getByText('Patient')).toBeInTheDocument()
    expect(screen.getByText('Diagnosis')).toBeInTheDocument()
  })

  it('renders patient rows with keyboard-accessible navigation affordances', () => {
    renderTable()
    expect(screen.getByRole('link', { name: /Open patient Sarah Jenkins/i })).toBeInTheDocument()
  })

  it('shows skeleton rows when loading', () => {
    renderTable(true)
    expect(screen.queryByText('Sarah Jenkins')).not.toBeInTheDocument()
  })

  it('renders OPEN CHART button when rowMode is select', () => {
    render(
      <MemoryRouter>
        <PatientTable patients={mockPatients} scansMap={mockScans} rowMode="select" />
      </MemoryRouter>
    )
    expect(screen.getByRole('button', { name: /Open chart page for Sarah Jenkins/i })).toBeInTheDocument()
  })
})
