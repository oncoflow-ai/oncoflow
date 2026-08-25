import { beforeEach, describe, expect, it, vi } from 'vitest'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import DoctorPatientDashboardPage from '@/pages/DoctorPatientDashboardPage'

const {
  getPatientMock,
  getScansMock,
  getSummaryMock,
  listReportsMock,
  generateReportMock,
  saveMriAnalysisReportMock,
  submitMriIngestionJobMock,
  submitDemoMriSegmentationJobMock,
  submitNiftiSegmentationJobMock,
  getJobStatusMock,
  getStudyResultsMock,
  listStudiesMock,
  submitComparisonMock,
  MockBackendApiError,
} = vi.hoisted(() => ({
  getPatientMock: vi.fn(),
  getScansMock: vi.fn(),
  getSummaryMock: vi.fn(),
  listReportsMock: vi.fn(),
  generateReportMock: vi.fn(),
  saveMriAnalysisReportMock: vi.fn(),
  submitMriIngestionJobMock: vi.fn(),
  submitDemoMriSegmentationJobMock: vi.fn(),
  submitNiftiSegmentationJobMock: vi.fn(),
  getJobStatusMock: vi.fn(),
  getStudyResultsMock: vi.fn(),
  listStudiesMock: vi.fn(),
  submitComparisonMock: vi.fn(),
  MockBackendApiError: class BackendApiError extends Error {},
}))

vi.mock('@/api/patients', () => ({ getPatient: getPatientMock }))
vi.mock('@/api/scans', () => ({ getScans: getScansMock }))
vi.mock('@/api/reports', () => ({
  getSummary: getSummaryMock,
  listReports: listReportsMock,
  generateReport: generateReportMock,
  saveMriAnalysisReport: saveMriAnalysisReportMock,
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

function renderPatientChart(tab = 'upload') {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: 0 },
      mutations: { retry: false },
    },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[`/doctor/patients/P-1001?tab=${tab}`]}>
        <Routes>
          <Route path="/doctor/patients/:id" element={<DoctorPatientDashboardPage />} />
          <Route path="/patients/:patientId/results/:studyId" element={<div>Clinical result: P-1001 / study-1</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

async function selectDicomFormat(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /DICOM Zip/i }))
}

async function selectClassDemoFormat(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /Class Demo Endpoint/i }))
}

describe('DoctorPatientDashboardPage MRI upload pipeline', () => {
  beforeEach(() => {
    vi.useRealTimers()
    getPatientMock.mockReset()
    getScansMock.mockReset()
    getSummaryMock.mockReset()
    listReportsMock.mockReset()
    generateReportMock.mockReset()
    saveMriAnalysisReportMock.mockReset()
    submitMriIngestionJobMock.mockReset()
    submitDemoMriSegmentationJobMock.mockReset()
    submitNiftiSegmentationJobMock.mockReset()
    getJobStatusMock.mockReset()
    getStudyResultsMock.mockReset()
    listStudiesMock.mockReset()
    submitComparisonMock.mockReset()

    getPatientMock.mockResolvedValue(DEMO_PATIENT)
    getScansMock.mockResolvedValue([])
    getSummaryMock.mockResolvedValue(null)
    listReportsMock.mockResolvedValue([])
    listStudiesMock.mockResolvedValue([])
  })

  it('renders the prefilled upload workspace for the selected patient', async () => {
    renderPatientChart()

    expect(await screen.findByText(/Upload MRI — segmentation pipeline/i)).toBeInTheDocument()
    expect(await screen.findByDisplayValue('P-1001 · Ada Lovelace')).toBeInTheDocument()
  })

  it('submits a NIfTI scan, mask, and acquisition date to the backend', async () => {
    submitNiftiSegmentationJobMock.mockResolvedValue({
      jobId: 'job-nifti-1', studyId: 'study-nifti-1', status: 'queued', stage: 'staged', submittedAt: '2026-04-12T11:24:03.996257Z',
    })
    getJobStatusMock.mockResolvedValue({
      jobId: 'job-nifti-1', studyId: 'study-nifti-1', status: 'completed', stage: 'completed', submittedAt: '2026-04-12T11:24:03.996257Z', error: null,
    })
    renderPatientChart()
    const user = userEvent.setup()
    const scan = new File(['nifti-bytes'], 't1c.nii.gz', { type: 'application/gzip' })
    const mask = new File(['mask-bytes'], 'mask.nii.gz', { type: 'application/gzip' })

    await user.upload(await screen.findByLabelText(/NIfTI Scan/i), scan)
    await user.upload(screen.getByLabelText(/Tumor Mask/i), mask)
    await user.clear(screen.getByLabelText('Source Label'))
    await user.type(screen.getByLabelText('Source Label'), 'Patient P01 - Baseline')
    await user.type(screen.getByLabelText('Acquisition Date'), '2024-01-15')
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    await waitFor(() => expect(submitNiftiSegmentationJobMock).toHaveBeenCalledTimes(1))
    expect(submitNiftiSegmentationJobMock).toHaveBeenCalledWith(expect.objectContaining({
      scanFile: scan, maskFile: mask, sourceLabel: 'Patient P01 - Baseline', acquiredAt: '2024-01-15',
    }))
  })

  it('submits the selected DICOM zip and source label to the backend', async () => {
    submitMriIngestionJobMock.mockResolvedValue({
      jobId: 'job-1', studyId: 'study-1', status: 'queued', stage: 'staged', submittedAt: '2026-04-12T11:24:03.996257Z',
    })
    getJobStatusMock.mockResolvedValue({
      jobId: 'job-1', studyId: 'study-1', status: 'failed', stage: 'profiling', submittedAt: '2026-04-12T11:24:03.996257Z',
      error: { code: 'ingestion-failed', message: 'mock failure', details: { stage: 'profiling' } },
    })
    renderPatientChart()
    const user = userEvent.setup()
    await selectDicomFormat(user)
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.clear(screen.getByLabelText('Source Label'))
    await user.type(screen.getByLabelText('Source Label'), 'local-demo')
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    await waitFor(() => expect(submitMriIngestionJobMock).toHaveBeenCalledWith(file, 'local-demo'))
  })

  it('submits class-demo uploads to the demo MRI endpoint', async () => {
    submitDemoMriSegmentationJobMock.mockResolvedValue({
      jobId: 'job-demo-1', studyId: 'study-demo-1', status: 'queued', stage: 'staged', submittedAt: '2026-04-12T11:24:03.996257Z',
    })
    getJobStatusMock.mockResolvedValue({
      jobId: 'job-demo-1', studyId: 'study-demo-1', status: 'completed', stage: 'completed', submittedAt: '2026-04-12T11:24:03.996257Z', error: null,
    })
    renderPatientChart()
    const user = userEvent.setup()
    await selectClassDemoFormat(user)
    const file = new File(['mri-body'], 'demo-scan.nii.gz', { type: 'application/gzip' })

    await user.upload(await screen.findByLabelText('MRI Upload'), file)
    await user.clear(screen.getByLabelText('Source Label'))
    await user.type(screen.getByLabelText('Source Label'), 'Class demo MRI')
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    await waitFor(() => expect(submitDemoMriSegmentationJobMock).toHaveBeenCalledWith({
      scanFile: file, sourceLabel: 'Class demo MRI', acquiredAt: '',
    }))
  })

  it('saves the report and opens a dedicated clinical result after a completed upload', async () => {
    submitMriIngestionJobMock.mockResolvedValue({
      jobId: 'job-1', studyId: 'study-1', status: 'queued', stage: 'staged', submittedAt: '2026-04-12T11:24:03.996257Z',
    })
    getJobStatusMock
      .mockResolvedValueOnce({ jobId: 'job-1', studyId: 'study-1', status: 'queued', stage: 'staged', submittedAt: '2026-04-12T11:24:03.996257Z', error: null })
      .mockResolvedValueOnce({ jobId: 'job-1', studyId: 'study-1', status: 'completed', stage: 'completed', submittedAt: '2026-04-12T11:24:03.996257Z', error: null })
    renderPatientChart()
    const user = userEvent.setup()
    await selectDicomFormat(user)
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    await waitFor(() => expect(getJobStatusMock.mock.calls.length).toBeGreaterThanOrEqual(2), { timeout: 10000 })
    expect(await screen.findByText('Clinical result: P-1001 / study-1')).toBeInTheDocument()
    expect(saveMriAnalysisReportMock).toHaveBeenCalledWith('P-1001', 'study-1')
  }, 10000)

  it('keeps failed-job messaging clinical and hides backend payload details', async () => {
    submitMriIngestionJobMock.mockResolvedValue({
      jobId: 'job-fail', studyId: 'study-fail', status: 'queued', stage: 'staged', submittedAt: '2026-04-12T11:24:03.996257Z',
    })
    getJobStatusMock.mockResolvedValue({
      jobId: 'job-fail', studyId: 'study-fail', status: 'failed', stage: 'infer', submittedAt: '2026-04-12T11:24:03.996257Z',
      error: {
        code: 'model-runtime-missing',
        message: 'ONCOFLOW_NNUNET_MODEL_DIR is required to enable real nnU-Net inference',
        details: { studyId: 'study-fail' },
      },
    })
    renderPatientChart()
    const user = userEvent.setup()
    await selectDicomFormat(user)
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    expect(await screen.findByText(/This analysis could not be completed/i)).toBeInTheDocument()
    expect(screen.queryByText(/ONCOFLOW_NNUNET_MODEL_DIR is required/)).not.toBeInTheDocument()
    expect(screen.queryByText(/model-runtime-missing/)).not.toBeInTheDocument()
  })

  it('defers result loading to the dedicated result page', async () => {
    submitMriIngestionJobMock.mockResolvedValue({
      jobId: 'job-404', studyId: 'study-1', status: 'queued', stage: 'staged', submittedAt: '2026-04-12T11:24:03.996257Z',
    })
    getJobStatusMock.mockResolvedValue({
      jobId: 'job-404', studyId: 'study-1', status: 'completed', stage: 'completed', submittedAt: '2026-04-12T11:24:03.996257Z', error: null,
    })
    renderPatientChart()
    const user = userEvent.setup()
    await selectDicomFormat(user)
    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })

    await user.upload(await screen.findByLabelText('MRI Study Zip'), file)
    await user.click(screen.getByRole('button', { name: 'Upload And Start' }))

    expect(await screen.findByText('Clinical result: P-1001 / study-1')).toBeInTheDocument()
    expect(getStudyResultsMock).not.toHaveBeenCalled()
  })
})
