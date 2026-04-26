import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import DashboardPage from '@/pages/DashboardPage'
import { useAuthStore } from '@/store/authStore'

const {
  getPatientsMock,
  getScansMock,
  submitMriIngestionJobMock,
  getJobStatusMock,
  getStudyResultsMock,
  MockBackendApiError,
} = vi.hoisted(() => {
  class HoistedBackendApiError extends Error {
    statusCode: number | null
    detail: unknown

    constructor(message: string, options?: { statusCode?: number | null; detail?: unknown }) {
      super(message)
      this.name = 'BackendApiError'
      this.statusCode = options?.statusCode ?? null
      this.detail = options?.detail ?? null
    }
  }

  return {
    getPatientsMock: vi.fn(),
    getScansMock: vi.fn(),
    submitMriIngestionJobMock: vi.fn(),
    getJobStatusMock: vi.fn(),
    getStudyResultsMock: vi.fn(),
    MockBackendApiError: HoistedBackendApiError,
  }
})

vi.mock('@/api/patients', () => ({
  getPatients: getPatientsMock,
}))

vi.mock('@/api/scans', () => ({
  getScans: getScansMock,
}))

vi.mock('@/api/backendWorkspace', () => ({
  BackendApiError: MockBackendApiError,
  submitMriIngestionJob: submitMriIngestionJobMock,
  getJobStatus: getJobStatusMock,
  getStudyResults: getStudyResultsMock,
}))

function renderDashboard() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
      mutations: { retry: false },
    },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <DashboardPage />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe('DashboardPage operator workspace', () => {
  beforeEach(() => {
    vi.useRealTimers()
    getPatientsMock.mockReset()
    getScansMock.mockReset()
    submitMriIngestionJobMock.mockReset()
    getJobStatusMock.mockReset()
    getStudyResultsMock.mockReset()

    getPatientsMock.mockResolvedValue([])
    getScansMock.mockResolvedValue([])

    useAuthStore.setState({
      physician: { id: 'DR-001', name: 'Dr. D. Cohen', initials: 'DC' },
    })
  })

  it('renders the upload workspace inside the dashboard', async () => {
    renderDashboard()

    expect(await screen.findByText('Live MRI backend test console')).toBeInTheDocument()
    expect(screen.getByLabelText('MRI Study Zip')).toBeInTheDocument()
    expect(screen.getByLabelText('Search patients')).toBeInTheDocument()
    expect(screen.getByText(/Mock roster/)).toBeInTheDocument()
  })

  it('submits the selected file and source label to the backend', async () => {
    submitMriIngestionJobMock.mockResolvedValue({
      jobId: 'job-1',
      studyId: 'study-1',
      status: 'queued',
      stage: 'staged',
      submittedAt: '2026-04-12T11:24:03.996257Z',
    })
    getJobStatusMock.mockResolvedValue({
      jobId: 'job-1',
      studyId: 'study-1',
      status: 'failed',
      stage: 'profiling',
      submittedAt: '2026-04-12T11:24:03.996257Z',
      error: {
        code: 'ingestion-failed',
        message: 'mock failure',
        details: { stage: 'profiling' },
      },
    })

    renderDashboard()
    const user = userEvent.setup()
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.type(screen.getByLabelText('Source Label'), 'local-demo')
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    await waitFor(() => {
      expect(submitMriIngestionJobMock).toHaveBeenCalledWith(file, 'local-demo')
    })
  })

  it('polls active jobs and loads results after completion', async () => {
    submitMriIngestionJobMock.mockResolvedValue({
      jobId: 'job-1',
      studyId: 'study-1',
      status: 'queued',
      stage: 'staged',
      submittedAt: '2026-04-12T11:24:03.996257Z',
    })

    getJobStatusMock
      .mockResolvedValueOnce({
        jobId: 'job-1',
        studyId: 'study-1',
        status: 'queued',
        stage: 'staged',
        submittedAt: '2026-04-12T11:24:03.996257Z',
        error: null,
      })
      .mockResolvedValueOnce({
        jobId: 'job-1',
        studyId: 'study-1',
        status: 'completed',
        stage: 'completed',
        submittedAt: '2026-04-12T11:24:03.996257Z',
        error: null,
      })

    getStudyResultsMock.mockResolvedValue({
      studyId: 'study-1',
      resultArtifact: {
        artifactKind: 'study-result-bundle',
        storageRoot: 'derived',
        relativePath: 'studies/study-1/results/study-result.json',
      },
      lesions: [
        {
          lesionId: 'lesion-001',
          boundingBox: { xMin: 1, yMin: 2, zMin: 3, xMax: 10, yMax: 11, zMax: 12 },
          measurements: { volumeMm3: 1234, longestDiameterMm: 18.5 },
          maskArtifact: {
            artifactKind: 'segmentation-mask',
            storageRoot: 'derived',
            relativePath: 'studies/study-1/lesions/component-001.nii.gz',
          },
          reviewArtifacts: [
            {
              artifactKind: 'review-overlay',
              storageRoot: 'derived',
              relativePath: 'studies/study-1/review/overlay-001.png',
            },
          ],
          metadata: null,
        },
      ],
      needsReview: true,
      caseQcReasons: ['selected canonical series do not share geometry'],
    })

    renderDashboard()
    const user = userEvent.setup()
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    await waitFor(() => expect(getJobStatusMock.mock.calls.length).toBeGreaterThanOrEqual(2), { timeout: 10000 })
    await waitFor(() => expect(getStudyResultsMock).toHaveBeenCalledWith('study-1'), { timeout: 10000 })

    expect(await screen.findByText('selected canonical series do not share geometry')).toBeInTheDocument()
    expect(screen.getByText('lesion-001')).toBeInTheDocument()
    expect(screen.getAllByText(/studies\/study-1\/lesions\/component-001\.nii\.gz/)).toHaveLength(2)
  }, 10000)

  it('renders backend failure details for failed jobs', async () => {
    submitMriIngestionJobMock.mockResolvedValue({
      jobId: 'job-fail',
      studyId: 'study-fail',
      status: 'queued',
      stage: 'staged',
      submittedAt: '2026-04-12T11:24:03.996257Z',
    })

    getJobStatusMock.mockResolvedValue({
      jobId: 'job-fail',
      studyId: 'study-fail',
      status: 'failed',
      stage: 'infer',
      submittedAt: '2026-04-12T11:24:03.996257Z',
      error: {
        code: 'model-runtime-missing',
        message: 'ONCOFLOW_NNUNET_MODEL_DIR is required to enable real nnU-Net inference',
        details: { studyId: 'study-fail' },
      },
    })

    renderDashboard()
    const user = userEvent.setup()
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    expect(await screen.findByText('Backend failure')).toBeInTheDocument()
    expect(screen.getAllByText(/ONCOFLOW_NNUNET_MODEL_DIR is required/)).toHaveLength(2)
    expect(screen.getAllByText(/model-runtime-missing/)).toHaveLength(2)
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

    renderDashboard()

    expect(await screen.findByText('Some scan histories could not be loaded.')).toBeInTheDocument()
    expect(await screen.findByText('Ada Lovelace')).toBeInTheDocument()
    expect(screen.getByText('Grace Hopper')).toBeInTheDocument()
  })

  it('shows a specific message when results are missing after completion', async () => {
    submitMriIngestionJobMock.mockResolvedValue({
      jobId: 'job-404',
      studyId: 'study-404',
      status: 'queued',
      stage: 'staged',
      submittedAt: '2026-04-12T11:24:03.996257Z',
    })

    getJobStatusMock.mockResolvedValue({
      jobId: 'job-404',
      studyId: 'study-404',
      status: 'completed',
      stage: 'completed',
      submittedAt: '2026-04-12T11:24:03.996257Z',
      error: null,
    })

    getStudyResultsMock.mockRejectedValue(
      new MockBackendApiError('result not found', {
        statusCode: 404,
        detail: 'result not found',
      })
    )

    renderDashboard()
    const user = userEvent.setup()
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    expect(
      await screen.findByText('Job completed, but the backend returned no stored results for this study.')
    ).toBeInTheDocument()
  })
})
