import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import RadiologistWorkspacePage from '@/pages/RadiologistWorkspacePage'
import { useAuthStore } from '@/store/authStore'

const {
  getPatientsMock,
  getScansMock,
  submitMriIngestionJobMock,
  submitDemoMriSegmentationJobMock,
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
    submitDemoMriSegmentationJobMock: vi.fn(),
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
  submitDemoMriSegmentationJob: submitDemoMriSegmentationJobMock,
  submitNiftiSegmentationJob: submitNiftiSegmentationJobMock,
  getJobStatus: getJobStatusMock,
  getStudyResults: getStudyResultsMock,
  listStudies: listStudiesMock,
  submitComparison: submitComparisonMock,
}))

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

function renderRadiologist() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
      mutations: { retry: false },
    },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={['/radiologist']}>
        <Routes>
          <Route path="/radiologist" element={<RadiologistWorkspacePage />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

async function selectAdaPatient(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole('button', { name: /Select patient Ada Lovelace/i }))
}

async function selectDicomFormat(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /DICOM Zip/i }))
}

async function selectClassDemoFormat(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /Single Scan/i }))
}

describe('RadiologistWorkspacePage', () => {
  beforeEach(() => {
    vi.useRealTimers()
    getPatientsMock.mockReset()
    getScansMock.mockReset()
    submitMriIngestionJobMock.mockReset()
    submitDemoMriSegmentationJobMock.mockReset()
    submitNiftiSegmentationJobMock.mockReset()
    getJobStatusMock.mockReset()
    getStudyResultsMock.mockReset()
    listStudiesMock.mockReset()
    submitComparisonMock.mockReset()

    getPatientsMock.mockResolvedValue([DEMO_PATIENT])
    getScansMock.mockResolvedValue([])
    listStudiesMock.mockResolvedValue([])

    useAuthStore.setState({
      user: { id: 'RAD-001', name: 'Alex Rahman', initials: 'AR', role: 'radiologist' },
    })
  })

  it('shows roster until a patient is selected', async () => {
    renderRadiologist()
    expect(await screen.findByText(/Choose a patient row/i)).toBeInTheDocument()
    expect(screen.queryByLabelText(/NIfTI Scan/i)).not.toBeInTheDocument()
  })

  it('shows upload workspace after selecting a patient', async () => {
    renderRadiologist()
    const user = userEvent.setup()
    await selectAdaPatient(user)
    expect(await screen.findByLabelText(/NIfTI Scan/i)).toBeInTheDocument()
    expect(screen.getByText(/Upload MRI — segmentation pipeline/i)).toBeInTheDocument()
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

    renderRadiologist()
    const user = userEvent.setup()
    await selectAdaPatient(user)
    const scan = new File(['nifti-bytes'], 't1c.nii.gz', { type: 'application/gzip' })
    const mask = new File(['mask-bytes'], 'mask.nii.gz', { type: 'application/gzip' })

    await user.upload(await screen.findByLabelText(/NIfTI Scan/i), scan)
    await user.upload(screen.getByLabelText(/Tumor Mask/i), mask)
    await user.clear(screen.getByLabelText('Source Label'))
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

    renderRadiologist()
    const user = userEvent.setup()
    await selectAdaPatient(user)
    await selectDicomFormat(user)
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.clear(screen.getByLabelText('Source Label'))
    await user.type(screen.getByLabelText('Source Label'), 'local-demo')
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    await waitFor(() => {
      expect(submitMriIngestionJobMock).toHaveBeenCalledWith(file, 'local-demo')
    })
  })

  it('submits class demo uploads to the demo MRI endpoint', async () => {
    submitDemoMriSegmentationJobMock.mockResolvedValue({
      jobId: 'job-demo-1',
      studyId: 'study-demo-1',
      status: 'queued',
      stage: 'staged',
      submittedAt: '2026-04-12T11:24:03.996257Z',
    })
    getJobStatusMock.mockResolvedValue({
      jobId: 'job-demo-1',
      studyId: 'study-demo-1',
      status: 'completed',
      stage: 'completed',
      submittedAt: '2026-04-12T11:24:03.996257Z',
      error: null,
    })
    getStudyResultsMock.mockResolvedValue({
      studyId: 'study-demo-1',
      resultArtifact: {
        artifactKind: 'study-result-bundle',
        storageRoot: 'derived',
        relativePath: 'studies/study-demo-1/results/study-result.json',
      },
      lesions: [],
      needsReview: false,
      caseQcReasons: [],
      metadata: {
        source: 'ground-truth-demo-mask',
      },
    })

    renderRadiologist()
    const user = userEvent.setup()
    await selectAdaPatient(user)
    await selectClassDemoFormat(user)
    const file = new File(['mri-body'], 'demo-scan.nii.gz', { type: 'application/gzip' })

    await user.upload(await screen.findByLabelText('MRI Upload'), file)
    await user.clear(screen.getByLabelText('Source Label'))
    await user.type(screen.getByLabelText('Source Label'), 'Class demo MRI')
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    await waitFor(() => {
      expect(submitDemoMriSegmentationJobMock).toHaveBeenCalledTimes(1)
    })
    expect(submitDemoMriSegmentationJobMock).toHaveBeenCalledWith({
      scanFile: file,
      sourceLabel: 'Class demo MRI',
      acquiredAt: '',
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

    renderRadiologist()
    const user = userEvent.setup()
    await selectAdaPatient(user)
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

    renderRadiologist()
    const user = userEvent.setup()
    await selectAdaPatient(user)
    await selectDicomFormat(user)
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    expect(await screen.findByText('Backend failure')).toBeInTheDocument()
    expect(screen.getAllByText(/ONCOFLOW_NNUNET_MODEL_DIR is required/)).toHaveLength(2)
    expect(screen.getAllByText(/model-runtime-missing/)).toHaveLength(2)
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

    renderRadiologist()
    const user = userEvent.setup()
    await selectAdaPatient(user)
    await selectDicomFormat(user)
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    expect(
      await screen.findByText('Job completed, but the backend returned no stored results for this study.')
    ).toBeInTheDocument()
  })
})

describe('RadiologistWorkspacePage longitudinal comparison panel', () => {
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

    getPatientsMock.mockResolvedValue([DEMO_PATIENT])
    getScansMock.mockResolvedValue([])
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

    useAuthStore.setState({
      user: { id: 'RAD-001', name: 'Alex Rahman', initials: 'AR', role: 'radiologist' },
    })
  })

  it('runs a comparison and renders growth metrics', async () => {
    renderRadiologist()
    const user = userEvent.setup()
    await selectAdaPatient(user)

    await waitFor(() => expect(listStudiesMock).toHaveBeenCalled())
    expect(await screen.findByText('Compare two scans, see tumor change')).toBeInTheDocument()

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
