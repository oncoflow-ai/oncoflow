import axios from 'axios'
import { apiClient } from './client'
import type {
  BackendBoundingBox3D,
  BackendCaseResult,
  BackendComparisonResponse,
  BackendJobStatusResponse,
  BackendJobSubmission,
  BackendStudyListItem,
} from '@/types'

export class BackendApiError extends Error {
  statusCode: number | null
  detail: unknown

  constructor(message: string, options?: { statusCode?: number | null; detail?: unknown }) {
    super(message)
    this.name = 'BackendApiError'
    this.statusCode = options?.statusCode ?? null
    this.detail = options?.detail ?? null
  }
}

function normalizeApiError(error: unknown, fallback: string): BackendApiError {
  if (axios.isAxiosError(error)) {
    const statusCode = error.response?.status ?? null
    const detail = error.response?.data?.detail ?? error.response?.data ?? null
    const message = typeof detail === 'string'
      ? detail
      : detail != null
        ? JSON.stringify(detail, null, 2)
        : error.message || fallback
    return new BackendApiError(message, { statusCode, detail })
  }

  if (error instanceof Error) {
    return new BackendApiError(error.message, { detail: error })
  }

  return new BackendApiError(fallback, { detail: error })
}

function normalizeBoundingBox(
  boundingBox: Record<string, number>
): BackendBoundingBox3D {
  return {
    xMin: boundingBox.xMin ?? boundingBox.x_min,
    xMax: boundingBox.xMax ?? boundingBox.x_max,
    yMin: boundingBox.yMin ?? boundingBox.y_min,
    yMax: boundingBox.yMax ?? boundingBox.y_max,
    zMin: boundingBox.zMin ?? boundingBox.z_min,
    zMax: boundingBox.zMax ?? boundingBox.z_max,
  }
}

function normalizeCaseResult(result: BackendCaseResult): BackendCaseResult {
  return {
    ...result,
    lesions: result.lesions.map(lesion => ({
      ...lesion,
      boundingBox: normalizeBoundingBox(lesion.boundingBox as Record<string, number>),
    })),
  }
}

export async function submitMriIngestionJob(
  file: File,
  sourceLabel?: string
): Promise<BackendJobSubmission> {
  const formData = new FormData()
  formData.append('study_archive', file)
  if (sourceLabel?.trim()) {
    formData.append('source_label', sourceLabel.trim())
  }

  try {
    const response = await apiClient.post<BackendJobSubmission>('/api/v1/jobs/mri-ingestion', formData)
    return response.data
  } catch (error) {
    throw normalizeApiError(error, 'Failed to submit MRI ingestion job')
  }
}

export async function getJobStatus(jobId: string): Promise<BackendJobStatusResponse> {
  try {
    const response = await apiClient.get<BackendJobStatusResponse>(`/api/v1/jobs/${jobId}`)
    return response.data
  } catch (error) {
    throw normalizeApiError(error, 'Failed to fetch job status')
  }
}

export async function getStudyResults(studyId: string): Promise<BackendCaseResult> {
  try {
    const response = await apiClient.get<BackendCaseResult>(`/api/v1/results/${studyId}`)
    return normalizeCaseResult(response.data)
  } catch (error) {
    throw normalizeApiError(error, 'Failed to fetch study results')
  }
}

export interface SubmitNiftiSegmentationJobInput {
  scanFile: File
  maskFile?: File | null
  sourceLabel?: string
  acquiredAt?: string
}

export async function submitNiftiSegmentationJob(
  input: SubmitNiftiSegmentationJobInput
): Promise<BackendJobSubmission> {
  const formData = new FormData()
  formData.append('scan_file', input.scanFile)
  if (input.maskFile) {
    formData.append('mask_file', input.maskFile)
  }
  if (input.sourceLabel?.trim()) {
    formData.append('source_label', input.sourceLabel.trim())
  }
  if (input.acquiredAt?.trim()) {
    formData.append('acquired_at', input.acquiredAt.trim())
  }

  try {
    const response = await apiClient.post<BackendJobSubmission>(
      '/api/v1/jobs/nifti-segmentation',
      formData
    )
    return response.data
  } catch (error) {
    throw normalizeApiError(error, 'Failed to submit NIfTI segmentation job')
  }
}

export interface SubmitDemoMriSegmentationJobInput {
  scanFile: File
  sourceLabel?: string
  acquiredAt?: string
  patientId?: string
}

export async function submitDemoMriSegmentationJob(
  input: SubmitDemoMriSegmentationJobInput
): Promise<BackendJobSubmission> {
  const formData = new FormData()
  formData.append('scan_file', input.scanFile)
  if (input.sourceLabel?.trim()) {
    formData.append('source_label', input.sourceLabel.trim())
  }
  if (input.acquiredAt?.trim()) {
    formData.append('acquired_at', input.acquiredAt.trim())
  }
  if (input.patientId?.trim()) {
    formData.append('patient_id', input.patientId.trim())
  }

  try {
    const response = await apiClient.post<BackendJobSubmission>(
      '/api/v1/jobs/demo-mri-segmentation',
      formData
    )
    return response.data
  } catch (error) {
    throw normalizeApiError(error, 'Failed to submit demo MRI segmentation job')
  }
}

export async function listStudies(): Promise<BackendStudyListItem[]> {
  try {
    const response = await apiClient.get<BackendStudyListItem[]>('/api/v1/results/studies')
    return response.data
  } catch (error) {
    throw normalizeApiError(error, 'Failed to list studies')
  }
}

export interface SubmitComparisonInput {
  baselineStudyId: string
  followupStudyId: string
}

export async function submitComparison(
  input: SubmitComparisonInput
): Promise<BackendComparisonResponse> {
  try {
    const response = await apiClient.post<BackendComparisonResponse>(
      '/api/v1/jobs/longitudinal-comparison',
      {
        baselineStudyId: input.baselineStudyId,
        followupStudyId: input.followupStudyId,
      }
    )
    return response.data
  } catch (error) {
    throw normalizeApiError(error, 'Failed to run longitudinal comparison')
  }
}
