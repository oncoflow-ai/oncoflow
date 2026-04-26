import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  getJobStatus,
  getStudyResults,
  submitMriIngestionJob,
} from '@/api/backendWorkspace'

const { postMock, getMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
  getMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    post: postMock,
    get: getMock,
  },
}))

describe('backend workspace api', () => {
  beforeEach(() => {
    postMock.mockReset()
    getMock.mockReset()
  })

  it('submits MRI ingestion with multipart form data', async () => {
    postMock.mockResolvedValue({
      data: {
        jobId: 'job-1',
        studyId: 'study-1',
        status: 'queued',
        stage: 'staged',
        submittedAt: '2026-04-12T11:24:03.996257Z',
      },
    })

    const file = new File(['zip-body'], 'exam-upload.zip', { type: 'application/zip' })
    const payload = await submitMriIngestionJob(file, 'local-demo')

    expect(payload.jobId).toBe('job-1')
    expect(postMock).toHaveBeenCalledTimes(1)
    expect(postMock.mock.calls[0][0]).toBe('/api/v1/jobs/mri-ingestion')
    expect(postMock.mock.calls[0][1]).toBeInstanceOf(FormData)

    const formData = postMock.mock.calls[0][1] as FormData
    expect(formData.get('study_archive')).toBe(file)
    expect(formData.get('source_label')).toBe('local-demo')
  })

  it('fetches job status from the live jobs endpoint', async () => {
    getMock.mockResolvedValue({
      data: {
        jobId: 'job-1',
        studyId: 'study-1',
        status: 'running',
        stage: 'infer',
        submittedAt: '2026-04-12T11:24:03.996257Z',
        error: null,
      },
    })

    const payload = await getJobStatus('job-1')

    expect(payload.stage).toBe('infer')
    expect(getMock).toHaveBeenCalledWith('/api/v1/jobs/job-1')
  })

  it('fetches stored case results from the results endpoint', async () => {
    getMock.mockResolvedValue({
      data: {
        studyId: 'study-1',
        resultArtifact: {
          artifactKind: 'study-result-bundle',
          storageRoot: 'derived',
          relativePath: 'studies/study-1/results/study-result.json',
        },
        lesions: [],
        needsReview: true,
        caseQcReasons: ['selected canonical series do not share geometry'],
      },
    })

    const payload = await getStudyResults('study-1')

    expect(payload.needsReview).toBe(true)
    expect(getMock).toHaveBeenCalledWith('/api/v1/results/study-1')
  })

  it('normalizes backend errors with status codes and detail payloads', async () => {
    getMock.mockRejectedValue({
      isAxiosError: true,
      message: 'Request failed with status code 404',
      response: {
        status: 404,
        data: { detail: 'result not found' },
      },
    })

    await expect(getStudyResults('missing')).rejects.toMatchObject({
      name: 'BackendApiError',
      statusCode: 404,
      message: 'result not found',
      detail: 'result not found',
    })
  })

  it('normalizes snake_case bounding boxes from the backend payload', async () => {
    getMock.mockResolvedValue({
      data: {
        studyId: 'study-1',
        resultArtifact: {
          artifactKind: 'study-result-bundle',
          storageRoot: 'derived',
          relativePath: 'studies/study-1/results/study-result.json',
        },
        lesions: [
          {
            lesionId: 'lesion-001',
            boundingBox: {
              x_min: 1,
              x_max: 10,
              y_min: 2,
              y_max: 11,
              z_min: 3,
              z_max: 12,
            },
            measurements: {
              volumeMm3: 1234,
              longestDiameterMm: 18.5,
            },
            maskArtifact: {
              artifactKind: 'segmentation-mask',
              storageRoot: 'derived',
              relativePath: 'studies/study-1/lesions/component-001.nii.gz',
            },
            reviewArtifacts: [],
            metadata: null,
          },
        ],
        needsReview: false,
        caseQcReasons: [],
      },
    })

    const payload = await getStudyResults('study-1')

    expect(payload.lesions[0].boundingBox).toEqual({
      xMin: 1,
      xMax: 10,
      yMin: 2,
      yMax: 11,
      zMin: 3,
      zMax: 12,
    })
  })

  it('stringifies structured backend error details', async () => {
    getMock.mockRejectedValue({
      isAxiosError: true,
      message: 'Request failed with status code 422',
      response: {
        status: 422,
        data: {
          detail: {
            field: 'studyArchive',
            message: 'must be a valid zip file',
          },
        },
      },
    })

    await expect(getStudyResults('broken')).rejects.toMatchObject({
      name: 'BackendApiError',
      statusCode: 422,
      message: JSON.stringify(
        {
          field: 'studyArchive',
          message: 'must be a valid zip file',
        },
        null,
        2
      ),
    })
  })

  it('falls back to the transport error message when detail is missing', async () => {
    getMock.mockRejectedValue({
      isAxiosError: true,
      message: 'Network Error',
      response: undefined,
    })

    await expect(getStudyResults('offline')).rejects.toMatchObject({
      name: 'BackendApiError',
      statusCode: null,
      message: 'Network Error',
    })
  })

  it('stringifies response payloads when the backend omits detail', async () => {
    getMock.mockRejectedValue({
      isAxiosError: true,
      message: 'Request failed with status code 503',
      response: {
        status: 503,
        data: {
          code: 'service-unavailable',
          retryAfterSeconds: 30,
        },
      },
    })

    await expect(getStudyResults('degraded')).rejects.toMatchObject({
      name: 'BackendApiError',
      statusCode: 503,
      message: JSON.stringify(
        {
          code: 'service-unavailable',
          retryAfterSeconds: 30,
        },
        null,
        2
      ),
      detail: {
        code: 'service-unavailable',
        retryAfterSeconds: 30,
      },
    })
  })
})
