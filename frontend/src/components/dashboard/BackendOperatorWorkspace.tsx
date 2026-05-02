import { useId, useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, CheckCircle2, Clock3, LoaderCircle, Upload, FileScan, ShieldAlert } from 'lucide-react'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import StatBlock from '@/components/shared/StatBlock'
import { formatVolume } from '@/lib/utils'
import {
  BackendApiError,
  getJobStatus,
  getStudyResults,
  submitMriIngestionJob,
  submitNiftiSegmentationJob,
} from '@/api/backendWorkspace'
import { cn } from '@/lib/utils'
import type {
  BackendArtifactRef,
  BackendCaseResult,
  BackendJobStatus,
  BackendJobStatusResponse,
  BackendJobSubmission,
} from '@/types'

const ACTIVE_STATUSES: BackendJobStatus[] = ['queued', 'running']

type ScanFormat = 'nifti' | 'dicom-zip'

function isNiftiFilename(name: string): boolean {
  const lower = name.toLowerCase()
  return lower.endsWith('.nii') || lower.endsWith('.nii.gz')
}

/**
 * Do not use accept=".nii,.nii.gz" — many OS / browser pickers classify
 * double-suffix files as .gz only, so the dialog greys out real .nii.gz
 * volumes. We omit `accept` for NIfTI and validate the filename on submit.
 */
const DICOM_ZIP_ACCEPT = '.zip,application/zip,application/x-zip-compressed,application/octet-stream'

function formatTimestamp(value: string): string {
  return new Date(value).toLocaleString('en-US', {
    month: 'short',
    day: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

function StatusBadge({ status }: { status: BackendJobStatus }) {
  const styles: Record<BackendJobStatus, string> = {
    queued: 'border-amber/30 bg-amber/10 text-amber',
    running: 'border-teal/30 bg-teal/10 text-teal',
    failed: 'border-danger/30 bg-danger/10 text-danger',
    completed: 'border-teal/20 bg-surface2 text-text1',
  }

  return (
    <span className={cn(
      'inline-flex items-center rounded-full border px-2.5 py-1 font-mono text-[10px] font-bold uppercase tracking-[0.18em]',
      styles[status]
    )}>
      {status}
    </span>
  )
}

function ArtifactPath({ artifact }: { artifact: BackendArtifactRef }) {
  return (
    <code className="block break-all rounded border border-border2 bg-surface2 px-3 py-2 text-[11px] text-text2">
      {artifact.storageRoot}/{artifact.relativePath}
    </code>
  )
}

function ResultSummary({ result }: { result: BackendCaseResult }) {
  return (
    <div className="grid gap-px border border-border bg-border lg:grid-cols-4">
      <StatBlock label="Study Result" value={result.studyId.slice(0, 8).toUpperCase()} />
      <StatBlock
        label="Lesions"
        value={result.lesions.length.toString()}
        badge={
          <span className="inline-flex items-center font-mono text-[10px] bg-surface2 text-text2 px-1.5 py-0.5">
            packaged results
          </span>
        }
      />
      <StatBlock
        label="Review State"
        value={result.needsReview ? 'FLAGGED' : 'CLEAR'}
        badge={
          <span className={cn(
            'inline-flex items-center font-mono text-[10px] px-1.5 py-0.5 uppercase tracking-widest',
            result.needsReview
              ? 'border border-amber/30 bg-amber/10 text-amber'
              : 'border border-teal/25 bg-teal/10 text-teal'
          )}>
            {result.needsReview ? 'needs review' : 'ready'}
          </span>
        }
      />
      <StatBlock
        label="QC Reasons"
        value={result.caseQcReasons.length.toString()}
        badge={
          <span className="inline-flex items-center font-mono text-[10px] bg-surface2 text-text2 px-1.5 py-0.5">
            case-level
          </span>
        }
      />
    </div>
  )
}

export interface BackendOperatorWorkspaceProps {
  headingEyebrow?: string
  headingTitle?: string
  headingDescription?: string
  /** When provided (e.g. patient context), replaces the source label field until edited elsewhere */
  prefilledSourceLabel?: string
  onJobReachedTerminal?: (payload: { studyId: string; status: 'completed' | 'failed' }) => void
}

export default function BackendOperatorWorkspace({
  headingEyebrow = 'Operator Workspace',
  headingTitle = 'Live MRI backend test console',
  headingDescription = 'Upload an MRI archive, follow backend processing stages in real time, and inspect case results, lesion packaging, review signals, and artifact lineage from the current backend.',
  prefilledSourceLabel,
  onJobReachedTerminal,
}: BackendOperatorWorkspaceProps = {}) {
  const inputId = useId()
  const maskInputId = useId()
  const queryClient = useQueryClient()
  const [scanFormat, setScanFormat] = useState<ScanFormat>('nifti')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [maskFile, setMaskFile] = useState<File | null>(null)
  const [sourceLabel, setSourceLabel] = useState('')
  const [acquiredAt, setAcquiredAt] = useState('')
  const [localError, setLocalError] = useState<string | null>(null)
  const [activeRun, setActiveRun] = useState<BackendJobSubmission | null>(null)
  const terminalHandledJobIdRef = useRef<string | null>(null)

  useEffect(() => {
    terminalHandledJobIdRef.current = null
  }, [activeRun?.jobId])

  useEffect(() => {
    if (prefilledSourceLabel === undefined) return
    setSourceLabel(prefilledSourceLabel)
  }, [prefilledSourceLabel])

  const dicomMutation = useMutation({
    mutationFn: ({ file, label }: { file: File; label?: string }) => submitMriIngestionJob(file, label),
    onMutate: () => {
      setLocalError(null)
    },
    onSuccess: run => {
      setActiveRun(run)
      queryClient.invalidateQueries({ queryKey: ['backend-studies'] })
    },
  })

  const niftiMutation = useMutation({
    mutationFn: ({
      scanFile,
      maskFile,
      sourceLabel,
      acquiredAt,
    }: {
      scanFile: File
      maskFile?: File | null
      sourceLabel?: string
      acquiredAt?: string
    }) =>
      submitNiftiSegmentationJob({
        scanFile,
        maskFile: maskFile ?? null,
        sourceLabel,
        acquiredAt,
      }),
    onMutate: () => {
      setLocalError(null)
    },
    onSuccess: run => {
      setActiveRun(run)
      queryClient.invalidateQueries({ queryKey: ['backend-studies'] })
    },
  })

  const submitMutation = scanFormat === 'nifti' ? niftiMutation : dicomMutation
  const isSubmitting = niftiMutation.isPending || dicomMutation.isPending

  const jobStatusQuery = useQuery({
    queryKey: ['backend-operator-job', activeRun?.jobId],
    queryFn: () => getJobStatus(activeRun!.jobId),
    enabled: !!activeRun?.jobId,
    refetchInterval: query => {
      const data = query.state.data as BackendJobStatusResponse | undefined
      return data && ACTIVE_STATUSES.includes(data.status) ? 1500 : false
    },
  })

  const resultQuery = useQuery({
    queryKey: ['backend-operator-result', activeRun?.studyId],
    queryFn: () => getStudyResults(activeRun!.studyId),
    enabled: !!activeRun?.studyId && jobStatusQuery.data?.status === 'completed',
  })

  const jobStatus = jobStatusQuery.data ?? activeRun
  const resultMissing =
    resultQuery.error instanceof BackendApiError && resultQuery.error.statusCode === 404
  const submissionError =
    submitMutation.error instanceof BackendApiError ? submitMutation.error : null
  const jobFetchError =
    jobStatusQuery.error instanceof BackendApiError ? jobStatusQuery.error : null

  useEffect(() => {
    const jobId = activeRun?.jobId
    const studyId = activeRun?.studyId
    const st = jobStatusQuery.data?.status
    if (!jobId || !studyId || !st || (st !== 'completed' && st !== 'failed')) return
    if (terminalHandledJobIdRef.current === jobId) return
    terminalHandledJobIdRef.current = jobId
    onJobReachedTerminal?.({ studyId, status: st })
  }, [activeRun?.jobId, activeRun?.studyId, jobStatusQuery.data?.status, onJobReachedTerminal])

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!selectedFile) {
      setLocalError(
        scanFormat === 'nifti'
          ? 'Select a NIfTI scan before submitting.'
          : 'Select a zipped MRI archive before submitting.'
      )
      return
    }

    if (scanFormat === 'nifti') {
      if (!isNiftiFilename(selectedFile.name)) {
        setLocalError('Scan must be a .nii or .nii.gz NIfTI volume.')
        return
      }
      if (maskFile && !isNiftiFilename(maskFile.name)) {
        setLocalError('Tumor mask must be a .nii or .nii.gz NIfTI volume.')
        return
      }
      if (acquiredAt && Number.isNaN(Date.parse(acquiredAt))) {
        setLocalError('Acquisition date must be a valid YYYY-MM-DD value.')
        return
      }
      niftiMutation.mutate({
        scanFile: selectedFile,
        maskFile,
        sourceLabel,
        acquiredAt,
      })
      return
    }

    if (!selectedFile.name.toLowerCase().endsWith('.zip')) {
      setLocalError('The workspace currently accepts .zip MRI study archives only.')
      return
    }

    dicomMutation.mutate({ file: selectedFile, label: sourceLabel })
  }

  return (
    <section className="border border-border bg-surface">
      <div className="border-b border-border px-5 py-4">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-teal">{headingEyebrow}</p>
            <h2 className="mt-1 text-[24px] font-sans font-bold text-text1">{headingTitle}</h2>
            <p className="mt-2 max-w-3xl text-[13px] leading-relaxed text-text2">
              {headingDescription}
            </p>
          </div>
          <div className="rounded border border-border2 bg-bg px-3 py-2 font-mono text-[11px] text-text3">
            POST <span className="text-text1">/api/v1/jobs/nifti-segmentation</span>
            <br />
            POST <span className="text-text1">/api/v1/jobs/mri-ingestion</span>
            <br />
            GET <span className="text-text1">/api/v1/jobs/&lt;jobId&gt;</span>
            <br />
            GET <span className="text-text1">/api/v1/results/&lt;studyId&gt;</span>
          </div>
        </div>
      </div>

      <div className="grid gap-px border-b border-border bg-border lg:grid-cols-[1.15fr,0.85fr]">
        <div className="bg-bg p-5">
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div className="flex items-center gap-2 text-[12px] font-mono uppercase tracking-[0.18em] text-text3">
              <Upload size={14} />
              Upload scan
            </div>

            <div>
              <p className="mb-2 block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                Scan Format
              </p>
              <div className="inline-flex border border-border2">
                <button
                  type="button"
                  onClick={() => {
                    setScanFormat('nifti')
                    setSelectedFile(null)
                  }}
                  className={cn(
                    'px-3 py-1.5 font-mono text-[11px] font-bold uppercase tracking-[0.18em]',
                    scanFormat === 'nifti'
                      ? 'bg-teal text-black'
                      : 'bg-surface text-text2 hover:text-text1'
                  )}
                >
                  NIfTI (recommended)
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setScanFormat('dicom-zip')
                    setSelectedFile(null)
                    setMaskFile(null)
                  }}
                  className={cn(
                    'border-l border-border2 px-3 py-1.5 font-mono text-[11px] font-bold uppercase tracking-[0.18em]',
                    scanFormat === 'dicom-zip'
                      ? 'bg-teal text-black'
                      : 'bg-surface text-text2 hover:text-text1'
                  )}
                >
                  DICOM Zip
                </button>
              </div>
            </div>

            <div>
              <label htmlFor={inputId} className="mb-2 block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                {scanFormat === 'nifti' ? 'NIfTI Scan (.nii / .nii.gz)' : 'MRI Study Zip'}
              </label>
              <input
                id={inputId}
                type="file"
                {...(scanFormat === 'nifti'
                  ? {}
                  : { accept: DICOM_ZIP_ACCEPT })}
                onChange={event => setSelectedFile(event.target.files?.[0] ?? null)}
                className="block w-full cursor-pointer border border-border2 bg-surface px-3 py-3 text-[13px] text-text2 file:mr-4 file:border-0 file:bg-teal file:px-3 file:py-1.5 file:font-mono file:text-[11px] file:font-bold file:uppercase file:tracking-widest file:text-black"
              />
              {selectedFile && (
                <p className="mt-2 text-[12px] text-text2">
                  Selected: <span className="font-mono text-text1">{selectedFile.name}</span>
                </p>
              )}
              {scanFormat === 'nifti' && (
                <p className="mt-2 text-[11px] text-text3">
                  The file dialog shows all files so <span className="font-mono">.nii.gz</span> volumes are never hidden by
                  the browser. Only <span className="font-mono">.nii</span> and <span className="font-mono">.nii.gz</span>{' '}
                  scan names pass validation.
                </p>
              )}
            </div>

            {scanFormat === 'nifti' && (
              <div>
                <label htmlFor={maskInputId} className="mb-2 block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                  Tumor Mask (optional, .nii.gz)
                </label>
                <input
                  id={maskInputId}
                  type="file"
                  onChange={event => setMaskFile(event.target.files?.[0] ?? null)}
                  className="block w-full cursor-pointer border border-border2 bg-surface px-3 py-3 text-[13px] text-text2 file:mr-4 file:border-0 file:bg-teal file:px-3 file:py-1.5 file:font-mono file:text-[11px] file:font-bold file:uppercase file:tracking-widest file:text-black"
                />
                {maskFile && (
                  <p className="mt-2 text-[12px] text-text2">
                    Mask: <span className="font-mono text-text1">{maskFile.name}</span>
                  </p>
                )}
                <p className="mt-2 text-[11px] text-text3">
                  When supplied, the mask is treated as the segmentation result (skips inference).
                  Accepted: plain <span className="font-mono">.nii</span> or{' '}
                  <span className="font-mono">.nii.gz</span> (filename is checked after you pick the file).
                </p>
              </div>
            )}

            <div className="grid gap-3 lg:grid-cols-2">
              <div>
                <label htmlFor="source-label" className="mb-2 block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                  Source Label
                </label>
                <input
                  id="source-label"
                  value={sourceLabel}
                  onChange={event => setSourceLabel(event.target.value)}
                  placeholder="Patient P01 - Baseline"
                  className="w-full border border-border2 bg-surface px-3.5 py-[10px] text-[14px] text-text1 placeholder-text3 focus:border-teal focus:outline-none"
                />
              </div>

              {scanFormat === 'nifti' && (
                <div>
                  <label htmlFor="acquired-at" className="mb-2 block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                    Acquisition Date
                  </label>
                  <input
                    id="acquired-at"
                    type="date"
                    value={acquiredAt}
                    onChange={event => setAcquiredAt(event.target.value)}
                    className="w-full border border-border2 bg-surface px-3.5 py-[10px] text-[14px] text-text1 focus:border-teal focus:outline-none"
                  />
                </div>
              )}
            </div>

            {localError && <ErrorBanner message={localError} />}
            {submissionError && <ErrorBanner message={submissionError.message} />}

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={isSubmitting}
                className="bg-teal px-4 py-2.5 font-mono text-[12px] font-bold uppercase tracking-[0.18em] text-black transition-colors hover:bg-teal/90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isSubmitting ? 'Submitting…' : 'Upload And Start'}
              </button>
              <span className="text-[12px] text-text3">
                The dashboard will poll automatically until the run reaches a terminal state.
              </span>
            </div>
          </form>
        </div>

        <div className="bg-bg p-5">
          <div className="mb-4 flex items-center gap-2 text-[12px] font-mono uppercase tracking-[0.18em] text-text3">
            <Clock3 size={14} />
            Active run status
          </div>

          {!jobStatus ? (
            <EmptyState
              icon={<FileScan size={24} />}
              title="No run submitted yet"
              description="Upload a NIfTI scan (and optional mask) or zipped DICOM study to start live backend tracking from this dashboard."
              className="min-h-[220px] border border-dashed border-border2 bg-surface"
            />
          ) : (
            <div className="space-y-4">
              <div className="rounded border border-border2 bg-surface p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Run status</p>
                    <p className="mt-1 text-[18px] font-sans font-semibold text-text1">{jobStatus.stage}</p>
                  </div>
                  {'status' in jobStatus && <StatusBadge status={jobStatus.status} />}
                </div>
                <dl className="grid gap-3 text-[12px] text-text2">
                  <div>
                    <dt className="font-mono uppercase tracking-[0.18em] text-text3">Job ID</dt>
                    <dd className="mt-1 break-all font-mono text-text1">{jobStatus.jobId}</dd>
                  </div>
                  <div>
                    <dt className="font-mono uppercase tracking-[0.18em] text-text3">Study ID</dt>
                    <dd className="mt-1 break-all font-mono text-text1">{jobStatus.studyId}</dd>
                  </div>
                  <div>
                    <dt className="font-mono uppercase tracking-[0.18em] text-text3">Submitted</dt>
                    <dd className="mt-1">{formatTimestamp(jobStatus.submittedAt)}</dd>
                  </div>
                </dl>
              </div>

              <div className="rounded border border-border2 bg-surface p-4">
                <div className="flex items-center gap-2 text-[12px] text-text2">
                  {jobStatusQuery.isFetching && ACTIVE_STATUSES.includes(jobStatus.status) ? (
                    <>
                      <LoaderCircle size={14} className="animate-spin text-teal" />
                      Polling backend for the latest stage update
                    </>
                  ) : jobStatus.status === 'completed' ? (
                    <>
                      <CheckCircle2 size={14} className="text-teal" />
                      Run completed. Loading results payload.
                    </>
                  ) : jobStatus.status === 'failed' ? (
                    <>
                      <AlertCircle size={14} className="text-danger" />
                      Run failed. Review the backend error payload below.
                    </>
                  ) : (
                    <>
                      <Clock3 size={14} className="text-amber" />
                      Waiting for the next backend transition.
                    </>
                  )}
                </div>
              </div>

              {jobFetchError && <ErrorBanner message={jobFetchError.message} />}
            </div>
          )}
        </div>
      </div>

      <div className="grid gap-px bg-border xl:grid-cols-[1.25fr,0.75fr]">
        <div className="bg-bg p-5">
          <div className="mb-4 flex items-center gap-2 text-[12px] font-mono uppercase tracking-[0.18em] text-text3">
            <ShieldAlert size={14} />
            Results and lesion packaging
          </div>

          {resultQuery.isLoading ? (
            <div className="border border-border2 bg-surface px-4 py-12 text-center text-[13px] text-text2">
              Fetching case results from the backend…
            </div>
          ) : resultMissing ? (
            <ErrorBanner message="Job completed, but the backend returned no stored results for this study." />
          ) : resultQuery.error instanceof BackendApiError ? (
            <ErrorBanner message={resultQuery.error.message} />
          ) : resultQuery.data ? (
            <div className="space-y-5">
              <ResultSummary result={resultQuery.data} />

              <div className="rounded border border-border2 bg-surface p-4">
                <p className="mb-2 text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Case QC reasons</p>
                {resultQuery.data.caseQcReasons.length > 0 ? (
                  <ul className="space-y-2 text-[13px] text-text2">
                    {resultQuery.data.caseQcReasons.map(reason => (
                      <li key={reason} className="rounded border border-amber/20 bg-amber/10 px-3 py-2">
                        {reason}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-[13px] text-text2">No case-level QC reasons returned for this run.</p>
                )}
              </div>

              <div className="rounded border border-border2 bg-surface p-4">
                <p className="mb-2 text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Result artifact</p>
                <ArtifactPath artifact={resultQuery.data.resultArtifact} />
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Lesion results</p>
                  <span className="text-[12px] text-text3">{resultQuery.data.lesions.length} lesion(s)</span>
                </div>

                {resultQuery.data.lesions.length === 0 ? (
                  <EmptyState
                    icon={<FileScan size={22} />}
                    title="No lesions returned"
                    description="This can happen for true empty cases, review-required studies, or when the configured model produced no components."
                    className="border border-border2 bg-surface"
                  />
                ) : (
                  <div className="space-y-3">
                    {resultQuery.data.lesions.map(lesion => (
                      <article key={lesion.lesionId} className="rounded border border-border2 bg-surface p-4">
                        <div className="mb-3 flex items-center justify-between gap-3">
                          <div>
                            <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Lesion ID</p>
                            <h3 className="mt-1 font-mono text-[16px] font-semibold text-text1">{lesion.lesionId}</h3>
                          </div>
                          <div className="text-right">
                            <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Measurements</p>
                            <p className="mt-1 text-[13px] text-text1">
                              {formatVolume(lesion.measurements.volumeMm3)} mm³ · {lesion.measurements.longestDiameterMm} mm
                            </p>
                          </div>
                        </div>

                        <div className="mb-3 grid gap-3 lg:grid-cols-2">
                          <div>
                            <p className="mb-2 text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Bounding box</p>
                            <code className="block rounded border border-border2 bg-bg px-3 py-2 text-[11px] text-text2">
                              {prettyJson(lesion.boundingBox)}
                            </code>
                          </div>
                          <div>
                            <p className="mb-2 text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Mask artifact</p>
                            <ArtifactPath artifact={lesion.maskArtifact} />
                          </div>
                        </div>

                        <div>
                          <p className="mb-2 text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Review artifacts</p>
                          {lesion.reviewArtifacts.length > 0 ? (
                            <div className="space-y-2">
                              {lesion.reviewArtifacts.map(artifact => (
                                <ArtifactPath
                                  key={`${artifact.storageRoot}/${artifact.relativePath}`}
                                  artifact={artifact}
                                />
                              ))}
                            </div>
                          ) : (
                            <p className="text-[12px] text-text2">No review artifacts returned for this lesion.</p>
                          )}
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <EmptyState
              icon={<CheckCircle2 size={24} />}
              title="Results will appear after a completed run"
              description="Once the backend finishes processing, this section will load case-level review flags and lesion outputs."
              className="min-h-[280px] border border-dashed border-border2 bg-surface"
            />
          )}
        </div>

        <div className="bg-bg p-5">
          <div className="mb-4 flex items-center gap-2 text-[12px] font-mono uppercase tracking-[0.18em] text-text3">
            <AlertCircle size={14} />
            Failure and raw payloads
          </div>

          <div className="space-y-4">
            {jobStatusQuery.data?.error ? (
              <div className="rounded border border-danger/30 bg-danger/10 p-4">
                <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-danger">Backend failure</p>
                <p className="mt-2 text-[14px] font-semibold text-danger">{jobStatusQuery.data.error.message}</p>
                <p className="mt-1 font-mono text-[12px] text-danger/90">Code: {jobStatusQuery.data.error.code}</p>
                {jobStatusQuery.data.error.details && (
                  <code className="mt-3 block whitespace-pre-wrap break-all rounded border border-danger/20 bg-bg px-3 py-2 text-[11px] text-text2">
                    {prettyJson(jobStatusQuery.data.error.details)}
                  </code>
                )}
              </div>
            ) : (
              <div className="rounded border border-border2 bg-surface p-4 text-[13px] text-text2">
                No backend failure payload for the current run.
              </div>
            )}

            <details className="rounded border border-border2 bg-surface p-4">
              <summary className="cursor-pointer text-[11px] font-mono uppercase tracking-[0.18em] text-text3">
                Raw job payload
              </summary>
              <pre className="mt-3 overflow-x-auto whitespace-pre-wrap break-words text-[11px] text-text2">
                {prettyJson(jobStatusQuery.data ?? activeRun ?? { status: 'idle' })}
              </pre>
            </details>

            <details className="rounded border border-border2 bg-surface p-4">
              <summary className="cursor-pointer text-[11px] font-mono uppercase tracking-[0.18em] text-text3">
                Raw results payload
              </summary>
              <pre className="mt-3 overflow-x-auto whitespace-pre-wrap break-words text-[11px] text-text2">
                {prettyJson(resultQuery.data ?? { status: resultMissing ? 'missing' : 'idle' })}
              </pre>
            </details>
          </div>
        </div>
      </div>
    </section>
  )
}
