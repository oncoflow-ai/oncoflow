import axios from 'axios'
import { apiClient } from './client'
import type {
  BackendBoundingBox3D,
  BackendCaseResult,
  BackendJobStatusResponse,
  BackendJobSubmission,
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
