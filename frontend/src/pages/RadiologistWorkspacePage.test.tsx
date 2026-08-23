import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import RadiologistWorkspacePage from '@/pages/RadiologistWorkspacePage'
import { useAuthStore } from '@/store/authStore'

const { getPatientsMock, getScansMock } = vi.hoisted(() => ({
  getPatientsMock: vi.fn(),
  getScansMock: vi.fn(),
}))

vi.mock('@/api/patients', () => ({ getPatients: getPatientsMock }))
vi.mock('@/api/scans', () => ({ getScans: getScansMock }))

const DEMO_PATIENT = {
  id: 'P-1001',
  name: 'Ada Lovelace',
  dob: '1985-01-01',
  diagnosis: 'Bone tumor',
  diagnosisLocation: 'Femur',
  assignedPhysicianId: 'DR-001',
  status: 'active' as const,
  scanCount: 2,
  lastScanDate: '2026-04-12',
}

function PatientChartLocation() {
  const location = useLocation()
  return <div>Patient chart: {location.pathname}{location.search}</div>
}

function renderRadiologist() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/radiologist']}>
        <Routes>
          <Route path="/radiologist" element={<RadiologistWorkspacePage />} />
          <Route path="/doctor/patients/:id" element={<PatientChartLocation />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('RadiologistWorkspacePage', () => {
  beforeEach(() => {
    getPatientsMock.mockReset()
    getScansMock.mockReset()
    getPatientsMock.mockResolvedValue([DEMO_PATIENT])
    getScansMock.mockResolvedValue([])
    useAuthStore.setState({
      user: { id: 'RAD-001', name: 'Alex Rahman', initials: 'AR', role: 'radiologist' },
    })
  })

  it('keeps the landing page roster-only and opens the selected patient Upload MRI chart', async () => {
    renderRadiologist()
    const user = userEvent.setup()

    await user.click(await screen.findByRole('button', { name: /Select patient Ada Lovelace/i }))

    expect(await screen.findByText('Patient chart: /doctor/patients/P-1001?tab=upload')).toBeInTheDocument()
    expect(screen.queryByText(/Upload MRI — segmentation pipeline/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Compare two scans, see tumor change/i)).not.toBeInTheDocument()
  })

  it('renders loading rows while patient data is loading', () => {
    getPatientsMock.mockImplementation(() => new Promise(() => {}))
    renderRadiologist()

    expect(screen.getAllByRole('row')).toHaveLength(6)
  })

  it('shows the patient load error banner', async () => {
    getPatientsMock.mockRejectedValue(new Error('offline'))
    renderRadiologist()

    expect(await screen.findByText('Failed to load patients.')).toBeInTheDocument()
  })

  it('shows the scan-history error banner while retaining the roster', async () => {
    getScansMock.mockRejectedValue(new Error('offline'))
    renderRadiologist()

    expect(await screen.findByText('Some scan histories could not be loaded.')).toBeInTheDocument()
    expect(screen.getByText('Ada Lovelace')).toBeInTheDocument()
  })

  it('shows an empty roster state when no patients are available', async () => {
    getPatientsMock.mockResolvedValue([])
    renderRadiologist()

    expect(await screen.findByText('No patients')).toBeInTheDocument()
  })
})
