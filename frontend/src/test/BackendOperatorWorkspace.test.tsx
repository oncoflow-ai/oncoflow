import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import BackendOperatorWorkspace from '@/components/dashboard/BackendOperatorWorkspace'
import { getJobStatus, submitNiftiSegmentationJob } from '@/api/backendWorkspace'

vi.mock('@/api/backendWorkspace', () => ({
  submitMriIngestionJob: vi.fn(),
  submitNiftiSegmentationJob: vi.fn(),
  submitDemoMriSegmentationJob: vi.fn(),
  getJobStatus: vi.fn(),
  BackendApiError: class extends Error {},
}))

function renderWorkspace() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <BackendOperatorWorkspace />
    </QueryClientProvider>
  )
}

describe('BackendOperatorWorkspace algorithm stage & progress tracking', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('displays algorithm pipeline stages and progress bar during active analysis', async () => {
    const user = userEvent.setup()
    vi.mocked(submitNiftiSegmentationJob).mockResolvedValue({
      jobId: 'job-progress-1',
      studyId: 'study-progress-1',
      status: 'running',
      stage: 'bone-extraction',
      progress: 35,
      stageMessage: 'Extracting bone structures and anatomical landmarks...',
      submittedAt: '2026-08-25T10:00:00Z',
    })

    vi.mocked(getJobStatus).mockResolvedValue({
      jobId: 'job-progress-1',
      studyId: 'study-progress-1',
      status: 'running',
      stage: 'bone-extraction',
      progress: 35,
      stageMessage: 'Extracting bone structures and anatomical landmarks...',
      submittedAt: '2026-08-25T10:00:00Z',
      error: null,
    })

    renderWorkspace()

    // Initially shows empty state
    expect(screen.getByText('No run submitted yet')).toBeInTheDocument()

    // Upload a NIfTI scan file
    const file = new File(['nifti-content'], 'brain_scan.nii.gz', { type: 'application/gzip' })
    const fileInput = screen.getByLabelText(/NIfTI Scan/i)
    await user.upload(fileInput, file)

    const startButton = screen.getByRole('button', { name: /Upload And Start/i })
    await user.click(startButton)

    // Verify progress percentage and stage message
    await waitFor(() => {
      expect(screen.getByText('35%')).toBeInTheDocument()
    })

    expect(screen.getByText(/Extracting bone structures and anatomical landmarks/i)).toBeInTheDocument()

    // Verify algorithm pipeline stages are rendered
    expect(screen.getByText('Data Ingestion & Loading')).toBeInTheDocument()
    expect(screen.getByText('Bone & Landmark Extraction')).toBeInTheDocument()
    expect(screen.getByText('AI Tumor Segmentation')).toBeInTheDocument()
    expect(screen.getByText('Volumetric Quantification')).toBeInTheDocument()
    expect(screen.getByText('Clinical Report Creation')).toBeInTheDocument()

    // Bone extraction is in progress
    expect(screen.getByText('IN PROGRESS')).toBeInTheDocument()
  })

  it('updates progress to 100% upon completion', async () => {
    const user = userEvent.setup()
    vi.mocked(submitNiftiSegmentationJob).mockResolvedValue({
      jobId: 'job-comp-1',
      studyId: 'study-comp-1',
      status: 'queued',
      stage: 'staged',
      progress: 0,
      stageMessage: 'Queued for analysis...',
      submittedAt: '2026-08-25T10:00:00Z',
    })

    vi.mocked(getJobStatus).mockResolvedValue({
      jobId: 'job-comp-1',
      studyId: 'study-comp-1',
      status: 'completed',
      stage: 'completed',
      progress: 100,
      stageMessage: 'Analysis completed successfully. Structured report and measurements are ready.',
      submittedAt: '2026-08-25T10:00:00Z',
      error: null,
    })

    renderWorkspace()

    const file = new File(['nifti-content'], 'brain_scan.nii.gz', { type: 'application/gzip' })
    const fileInput = screen.getByLabelText(/NIfTI Scan/i)
    await user.upload(fileInput, file)

    await user.click(screen.getByRole('button', { name: /Upload And Start/i }))

    await waitFor(() => {
      expect(screen.getByText('100%')).toBeInTheDocument()
    })
    expect(screen.getByText(/Analysis completed successfully/i)).toBeInTheDocument()
  })
})
