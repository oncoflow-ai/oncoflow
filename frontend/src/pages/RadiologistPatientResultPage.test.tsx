import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import RadiologistPatientResultPage from '@/pages/RadiologistPatientResultPage'
import { getPatient } from '@/api/patients'
import { getStudyResults } from '@/api/backendWorkspace'
import { useAuthStore } from '@/store/authStore'

vi.mock('@/api/patients', () => ({ getPatient: vi.fn() }))
vi.mock('@/api/backendWorkspace', () => ({
  getStudyResults: vi.fn(),
  getArtifactUrl: vi.fn((path: string) => `/api/v1/artifacts/${path}`),
}))

const DEMO_PATIENT = {
  id: 'P-9001',
  name: 'Demo Patient P01',
  dob: '1975-06-01',
  diagnosis: 'Demo lesion (sample BraTS volumes)',
  diagnosisLocation: 'See repo data/P01',
  assignedPhysicianId: 'DR-001',
  status: 'active' as const,
  scanCount: 1,
  lastScanDate: '2026-04-12',
}

const DEMO_RESULT = {
  studyId: 'study-demo-1',
  needsReview: false,
  caseQcReasons: [],
  resultArtifact: {
    artifactKind: 'result',
    storageRoot: 'local',
    relativePath: 'results/study-demo-1.json',
  },
  lesions: [
    {
      lesionId: 'lesion-001',
      boundingBox: { xMin: 0, xMax: 10, yMin: 0, yMax: 10, zMin: 0, zMax: 10 },
      measurements: {
        volumeMm3: 14815,
        longestDiameterMm: 64.8,
      },
      maskArtifact: {
        artifactKind: 'mask',
        storageRoot: 'local',
        relativePath: 'masks/mask.nii.gz',
      },
      reviewArtifacts: [],
    },
  ],
  metadata: {
    report: {
      title: 'AI brain MRI segmentation report',
      technique: 'Automated volumetric tumor segmentation',
      findings: 'Solitary enhancing intra-axial mass',
      comparison: 'Compared with previous scan',
      impression: 'Mild interval progression',
      recommendations: ['Follow-up in 3 months'],
    },
  },
}

function Destination() {
  const location = useLocation()
  return <div>Destination: {location.pathname}{location.search}</div>
}

function renderResultPage(patientId = 'P-9001', studyId = 'study-demo-1') {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 0 } },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[`/radiologist/patients/${patientId}/results/${studyId}`]}>
        <Routes>
          <Route
            path="/radiologist/patients/:patientId/results/:studyId"
            element={<RadiologistPatientResultPage />}
          />
          <Route path="/doctor/patients/:id" element={<Destination />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('RadiologistPatientResultPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(getPatient).mockResolvedValue(DEMO_PATIENT)
    vi.mocked(getStudyResults).mockResolvedValue(DEMO_RESULT)
    useAuthStore.setState({
      user: { id: 'RAD-001', name: 'Alex Rahman', initials: 'AR', role: 'radiologist' },
    })
  })

  it('clicking Upload another MRI navigates back to the specific patient upload tab', async () => {
    renderResultPage('P-9001', 'study-demo-1')
    const user = userEvent.setup()

    expect(await screen.findByText('Segmentation result')).toBeInTheDocument()
    expect(await screen.findByText(/Demo Patient P01/i)).toBeInTheDocument()

    const backButton = screen.getByRole('link', { name: /Upload another MRI/i })
    expect(backButton).toHaveAttribute('href', '/doctor/patients/P-9001?tab=upload')

    await user.click(backButton)

    expect(await screen.findByText('Destination: /doctor/patients/P-9001?tab=upload')).toBeInTheDocument()
  })
})
