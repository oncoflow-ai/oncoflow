import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { render, screen } from '@testing-library/react'
import DoctorDashboardPage from '@/pages/DoctorDashboardPage'
import { useAuthStore } from '@/store/authStore'

const { getPatientsMock, getScansMock } = vi.hoisted(() => ({
  getPatientsMock: vi.fn(),
  getScansMock: vi.fn(),
}))

vi.mock('@/api/patients', () => ({
  getPatients: getPatientsMock,
  createPatient: vi.fn(),
}))

vi.mock('@/api/scans', () => ({
  getScans: getScansMock,
}))

function renderDoctor() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
      mutations: { retry: false },
    },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/doctor']}>
        <Routes>
          <Route path="/doctor" element={<DoctorDashboardPage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('DoctorDashboardPage', () => {
  beforeEach(() => {
    getPatientsMock.mockReset()
    getScansMock.mockReset()
    useAuthStore.setState({
      user: { id: 'DR-001', name: 'Dr. D. Cohen', initials: 'DC', role: 'doctor' },
    })
  })

  it('keeps patient rows visible when one scan request fails', async () => {
    getPatientsMock.mockResolvedValue([
      {
        id: 'P-1001',
        name: 'Ada Lovelace',
        dob: '1985-01-01',
        diagnosis: 'Bone tumor',
        diagnosisLocation: 'Femur',
        assignedPhysicianId: 'DR-001',
        status: 'active',
        scanCount: 2,
        lastScanDate: '2026-04-12',
      },
      {
        id: 'P-1002',
        name: 'Grace Hopper',
        dob: '1984-02-02',
        diagnosis: 'Follow-up',
        diagnosisLocation: 'Pelvis',
        assignedPhysicianId: 'DR-001',
        status: 'review',
        scanCount: 1,
        lastScanDate: '2026-04-11',
      },
    ])

    getScansMock
      .mockResolvedValueOnce([
        {
          id: 'SCN-1',
          patientId: 'P-1001',
          studyLabel: 'MRI Study #1',
          date: '2026-04-12',
          modality: 'MRI',
          sequence: 'T1W',
          plane: 'AXIAL',
          sliceCount: 120,
          resolution: '1.2mm iso',
          volumeMm3: 1200,
          maxDiameterMm: 18,
          isAnnotated: true,
        },
      ])
      .mockRejectedValueOnce(new Error('scan service unavailable'))

    renderDoctor()

    expect(await screen.findByText('Some scan histories could not be loaded.')).toBeInTheDocument()
    expect(await screen.findByText('Ada Lovelace')).toBeInTheDocument()
    expect(screen.getByText('Grace Hopper')).toBeInTheDocument()
  })
})
