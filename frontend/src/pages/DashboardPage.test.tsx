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
  submitNiftiSegmentationJobMock,
  getJobStatusMock,
  getStudyResultsMock,
  listStudiesMock,
  submitComparisonMock,
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
    submitNiftiSegmentationJobMock: vi.fn(),
    getJobStatusMock: vi.fn(),
    getStudyResultsMock: vi.fn(),
    listStudiesMock: vi.fn(),
    submitComparisonMock: vi.fn(),
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
  submitNiftiSegmentationJob: submitNiftiSegmentationJobMock,
  getJobStatus: getJobStatusMock,
  getStudyResults: getStudyResultsMock,
  listStudies: listStudiesMock,
  submitComparison: submitComparisonMock,
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

async function selectDicomFormat(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /DICOM Zip/i }))
}

describe('DashboardPage operator workspace', () => {
  beforeEach(() => {
    vi.useRealTimers()
    getPatientsMock.mockReset()
    getScansMock.mockReset()
    submitMriIngestionJobMock.mockReset()
    submitNiftiSegmentationJobMock.mockReset()
    getJobStatusMock.mockReset()
    getStudyResultsMock.mockReset()
    listStudiesMock.mockReset()
    submitComparisonMock.mockReset()

    getPatientsMock.mockResolvedValue([])
    getScansMock.mockResolvedValue([])
    listStudiesMock.mockResolvedValue([])

    useAuthStore.setState({
      physician: { id: 'DR-001', name: 'Dr. D. Cohen', initials: 'DC' },
    })
  })

  it('renders the upload workspace inside the dashboard', async () => {
    renderDashboard()

    expect(await screen.findByText('Live MRI backend test console')).toBeInTheDocument()
    expect(screen.getByLabelText(/NIfTI Scan/i)).toBeInTheDocument()
    expect(screen.getByLabelText('Search patients')).toBeInTheDocument()
    expect(screen.getByText(/Mock roster/)).toBeInTheDocument()
  })

  it('submits a NIfTI scan + mask + acquisition date to the backend', async () => {
    submitNiftiSegmentationJobMock.mockResolvedValue({
      jobId: 'job-nifti-1',
      studyId: 'study-nifti-1',
      status: 'queued',
      stage: 'staged',
      submittedAt: '2026-04-12T11:24:03.996257Z',
    })
    getJobStatusMock.mockResolvedValue({
      jobId: 'job-nifti-1',
      studyId: 'study-nifti-1',
      status: 'completed',
      stage: 'completed',
      submittedAt: '2026-04-12T11:24:03.996257Z',
      error: null,
    })
    getStudyResultsMock.mockResolvedValue({
      studyId: 'study-nifti-1',
      resultArtifact: {
        artifactKind: 'study-result-bundle',
        storageRoot: 'derived',
        relativePath: 'studies/study-nifti-1/results/study-result.json',
      },
      lesions: [],
      needsReview: false,
      caseQcReasons: [],
    })

    renderDashboard()
    const user = userEvent.setup()
    const scan = new File(['nifti-bytes'], 't1c.nii.gz', { type: 'application/gzip' })
    const mask = new File(['mask-bytes'], 'mask.nii.gz', { type: 'application/gzip' })

    await user.upload(await screen.findByLabelText(/NIfTI Scan/i), scan)
    await user.upload(screen.getByLabelText(/Tumor Mask/i), mask)
    await user.type(screen.getByLabelText('Source Label'), 'Patient P01 - Baseline')
    await user.type(screen.getByLabelText('Acquisition Date'), '2024-01-15')
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    await waitFor(() => {
      expect(submitNiftiSegmentationJobMock).toHaveBeenCalledTimes(1)
    })
    const payload = submitNiftiSegmentationJobMock.mock.calls[0][0]
    expect(payload.scanFile).toBe(scan)
    expect(payload.maskFile).toBe(mask)
    expect(payload.sourceLabel).toBe('Patient P01 - Baseline')
    expect(payload.acquiredAt).toBe('2024-01-15')
  })

  it('submits the selected DICOM zip and source label to the backend', async () => {
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
    await selectDicomFormat(user)
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
    await selectDicomFormat(user)
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
    await selectDicomFormat(user)
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
    await selectDicomFormat(user)
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    expect(
      await screen.findByText('Job completed, but the backend returned no stored results for this study.')
    ).toBeInTheDocument()
  })
})

describe('DashboardPage longitudinal comparison panel', () => {
  beforeEach(() => {
    vi.useRealTimers()
    getPatientsMock.mockReset()
    getScansMock.mockReset()
    submitMriIngestionJobMock.mockReset()
    submitNiftiSegmentationJobMock.mockReset()
    getJobStatusMock.mockReset()
    getStudyResultsMock.mockReset()
    listStudiesMock.mockReset()
    submitComparisonMock.mockReset()

    getPatientsMock.mockResolvedValue([])
    getScansMock.mockResolvedValue([])

    useAuthStore.setState({
      physician: { id: 'DR-001', name: 'Dr. D. Cohen', initials: 'DC' },
    })
  })

  it('runs a comparison and renders growth metrics', async () => {
    listStudiesMock.mockResolvedValue([
      {
        studyId: 'baseline-study-id',
        sourceKind: 'nifti-upload',
        sourceLabel: 'Patient P01 - Baseline',
        acquiredAt: '2024-01-15',
        createdAt: '2024-01-15T10:00:00Z',
        jobStatus: 'completed',
        hasResults: true,
      },
      {
        studyId: 'fu1-study-id',
        sourceKind: 'nifti-upload',
        sourceLabel: 'Patient P01 - FU1',
        acquiredAt: '2024-04-10',
        createdAt: '2024-04-10T10:00:00Z',
        jobStatus: 'completed',
        hasResults: true,
      },
    ])

    submitComparisonMock.mockResolvedValue({
      comparisonId: 'cmp-001',
      baselineStudyId: 'baseline-study-id',
      followupStudyId: 'fu1-study-id',
      baselineAcquiredAt: '2024-01-15',
      followupAcquiredAt: '2024-04-10',
      metrics: {
        volumeACm3: 14.815,
        volumeBCm3: 18.5,
        deltaCm3: 3.685,
        pctChange: 24.87,
        diceOverlap: 0.78,
        hd95Mm: 4.2,
        recistAMm: 39.1,
        recistBMm: 43.2,
        recistRatio: 1.105,
        growthRateCm3PerDay: 0.043,
        registrationNcc: 0.97,
        volDeltaCiHalfCm3: 0.21,
        method: 'affine',
        backend: 'sitk',
        didResegment: false,
      },
      interpretation: 'Progressive disease',
      notes: [],
      outputRelativePath: 'comparisons/cmp-001',
    })

    renderDashboard()
    const user = userEvent.setup()

    await waitFor(() => expect(listStudiesMock).toHaveBeenCalled())
    expect(
      await screen.findByText('Compare two scans, see tumor change')
    ).toBeInTheDocument()

    const baselineSelect = await screen.findByLabelText('Baseline Study')
    const followupSelect = screen.getByLabelText('Follow-up Study')

    await user.selectOptions(baselineSelect, 'baseline-study-id')
    await user.selectOptions(followupSelect, 'fu1-study-id')
    await user.click(screen.getByRole('button', { name: 'Run Comparison' }))

    await waitFor(() => {
      expect(submitComparisonMock).toHaveBeenCalledWith({
        baselineStudyId: 'baseline-study-id',
        followupStudyId: 'fu1-study-id',
      })
    })

    expect(await screen.findByText('cmp-001', undefined, { timeout: 5000 })).toBeInTheDocument()
    expect(screen.getByText('14.81 cm³')).toBeInTheDocument()
    expect(screen.getByText('18.50 cm³')).toBeInTheDocument()
    expect(screen.getByText('+3.69 cm³')).toBeInTheDocument()
    expect(screen.getByText('+24.9%')).toBeInTheDocument()
    expect(screen.getByText('Progressive disease')).toBeInTheDocument()
  })
})
