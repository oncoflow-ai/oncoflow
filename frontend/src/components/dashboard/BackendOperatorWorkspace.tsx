import { useId, useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, CheckCircle2, Clock3, LoaderCircle, Upload, FileScan } from 'lucide-react'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import {
  BackendApiError,
  getJobStatus,
  submitDemoMriSegmentationJob,
  submitMriIngestionJob,
  submitNiftiSegmentationJob,
} from '@/api/backendWorkspace'
import { cn } from '@/lib/utils'
import type { BackendJobStatus, BackendJobStatusResponse, BackendJobSubmission } from '@/types'

const ACTIVE_STATUSES: BackendJobStatus[] = ['queued', 'running']

type ScanFormat = 'nifti' | 'dicom-zip' | 'class-demo'

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

const ALGORITHM_STAGES: Array<{
  id: string
  stepNumber: number
  title: string
  description: string
  minProgress: number
  matchingStages: string[]
}> = [
  {
    id: 'data-fetching',
    stepNumber: 1,
    title: 'Data Ingestion & Loading',
    description: 'Validating MRI volume format, headers, and preparing tensor inputs',
    minProgress: 15,
    matchingStages: ['staged', 'profiling', 'data-fetching', 'prepare-inputs'],
  },
  {
    id: 'bone-extraction',
    stepNumber: 2,
    title: 'Bone & Landmark Extraction',
    description: 'Extracting skeletal boundaries, bone contours, and spatial landmarks',
    minProgress: 35,
    matchingStages: ['bone-extraction'],
  },
  {
    id: 'segmentation',
    stepNumber: 3,
    title: 'AI Tumor Segmentation',
    description: 'Running deep learning multi-planar tumor segmentation models',
    minProgress: 65,
    matchingStages: ['segmentation', 'infer', 'demo-inference'],
  },
  {
    id: 'quantification',
    stepNumber: 4,
    title: 'Volumetric Quantification',
    description: 'Calculating tumor volume (mm³), max axial diameter, and lesion extent',
    minProgress: 80,
    matchingStages: ['quantification', 'postprocess', 'package-results'],
  },
  {
    id: 'report-generation',
    stepNumber: 5,
    title: 'Clinical Report Creation',
    description: 'Synthesizing structured findings, recommendations, and artifacts',
    minProgress: 95,
    matchingStages: ['report-generation', 'materialize-results'],
  },
]

export interface BackendOperatorWorkspaceProps {
  headingEyebrow?: string
  headingTitle?: string
  headingDescription?: string
  /** When provided (e.g. patient context), replaces the source label field until edited elsewhere */
  prefilledSourceLabel?: string
  patientId?: string
  onJobReachedTerminal?: (payload: {
    studyId: string
    status: 'completed' | 'failed'
    mode: ScanFormat
  }) => void
}

export default function BackendOperatorWorkspace({
  headingEyebrow = 'Operator Workspace',
  headingTitle = 'Upload MRI for analysis',
  headingDescription = 'Upload an MRI study for segmentation. When processing is complete, the clinical result opens automatically and is saved to the patient report history.',
  prefilledSourceLabel,
  patientId,
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
  const [activeRunMode, setActiveRunMode] = useState<ScanFormat>('nifti')
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
      setActiveRunMode('dicom-zip')
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
      setActiveRunMode('nifti')
      queryClient.invalidateQueries({ queryKey: ['backend-studies'] })
    },
  })

  const demoMutation = useMutation({
    mutationFn: ({
      scanFile,
      sourceLabel,
      acquiredAt,
      patientId: targetPatientId,
    }: {
      scanFile: File
      sourceLabel?: string
      acquiredAt?: string
      patientId?: string
    }) =>
      submitDemoMriSegmentationJob({
        scanFile,
        sourceLabel,
        acquiredAt,
        patientId: targetPatientId,
      }),
    onMutate: () => {
      setLocalError(null)
    },
    onSuccess: run => {
      setActiveRun(run)
      setActiveRunMode('class-demo')
      queryClient.invalidateQueries({ queryKey: ['backend-studies'] })
    },
  })

  const submitMutation =
    scanFormat === 'nifti'
      ? niftiMutation
      : scanFormat === 'class-demo'
        ? demoMutation
        : dicomMutation
  const isSubmitting = niftiMutation.isPending || dicomMutation.isPending || demoMutation.isPending

  const jobStatusQuery = useQuery({
    queryKey: ['backend-operator-job', activeRun?.jobId],
    queryFn: () => getJobStatus(activeRun!.jobId),
    enabled: !!activeRun?.jobId,
    refetchInterval: query => {
      const data = query.state.data as BackendJobStatusResponse | undefined
      return data && ACTIVE_STATUSES.includes(data.status) ? 800 : false
    },
  })

  const jobStatus = jobStatusQuery.data ?? activeRun
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
    onJobReachedTerminal?.({ studyId, status: st, mode: activeRunMode })
  }, [activeRun?.jobId, activeRun?.studyId, activeRunMode, jobStatusQuery.data?.status, onJobReachedTerminal])

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
        patientId,
        acquiredAt,
      })
      return
    }

    if (scanFormat === 'class-demo') {
      if (acquiredAt && Number.isNaN(Date.parse(acquiredAt))) {
        setLocalError('Acquisition date must be a valid YYYY-MM-DD value.')
        return
      }
      demoMutation.mutate({
        scanFile: selectedFile,
        sourceLabel,
        acquiredAt,
        patientId,
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
        <div className="flex flex-col gap-2">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-teal">{headingEyebrow}</p>
            <h2 className="mt-1 text-[24px] font-sans font-bold text-text1">{headingTitle}</h2>
            <p className="mt-2 max-w-3xl text-[13px] leading-relaxed text-text2">
              {headingDescription}
            </p>
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
                    setScanFormat('class-demo')
                    setSelectedFile(null)
                    setMaskFile(null)
                  }}
                  className={cn(
                    'border-l border-border2 px-3 py-1.5 font-mono text-[11px] font-bold uppercase tracking-[0.18em]',
                    scanFormat === 'class-demo'
                      ? 'bg-teal text-black'
                      : 'bg-surface text-text2 hover:text-text1'
                  )}
                >
                  Class Demo Endpoint
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
              {scanFormat === 'class-demo' && (
                <div className="mt-3 border border-teal/25 bg-teal/5 px-3 py-2 text-[12px] leading-relaxed text-text2">
                  Single-scan analysis mode. Upload an MRI study to run segmentation and generate a structured result.
                </div>
              )}
            </div>

            <div>
              <label htmlFor={inputId} className="mb-2 block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                {scanFormat === 'dicom-zip'
                  ? 'MRI Study Zip'
                  : scanFormat === 'class-demo'
                    ? 'MRI Upload'
                    : 'NIfTI Scan (.nii / .nii.gz)'}
              </label>
              <input
                id={inputId}
                type="file"
                {...(scanFormat === 'dicom-zip'
                  ? { accept: DICOM_ZIP_ACCEPT }
                  : {})}
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
              {scanFormat === 'class-demo' && (
                <p className="mt-2 text-[11px] text-text3">
                  Upload an MRI study file to generate a single-scan segmentation result.
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

              {(scanFormat === 'nifti' || scanFormat === 'class-demo') && (
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
            {submissionError && <ErrorBanner message="We couldn't start this analysis. Please try the upload again." />}

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
          ) : (() => {
            const currentProgress = jobStatus.status === 'completed'
              ? 100
              : Math.min(100, Math.max(0, jobStatus.progress ?? (jobStatus.status === 'queued' ? 0 : 15)))

            const currentMessage = jobStatus.status === 'failed'
              ? 'This analysis could not be completed. Please try again or contact support.'
              : (jobStatus.stageMessage || (
                jobStatus.status === 'completed'
                  ? 'Analysis completed successfully. Structured report and measurements are ready.'
                  : 'Algorithm running on MRI scan...'
              ))

            return (
              <div className="space-y-4">
                {/* Header & Status */}
                <div className="rounded border border-border2 bg-surface p-4">
                  <div className="mb-3 flex items-center justify-between gap-3">
                    <div>
                      <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Algorithm Status</p>
                      <p className="mt-1 text-[16px] font-sans font-semibold text-text1 capitalize">{jobStatus.stage.replace(/-/g, ' ')}</p>
                    </div>
                    {'status' in jobStatus && <StatusBadge status={jobStatus.status} />}
                  </div>

                  {/* Percentage Progress Bar */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-[11px] font-mono">
                      <span className="text-text3 uppercase tracking-wider">Progress</span>
                      <span className="font-bold text-teal">{currentProgress}%</span>
                    </div>
                    <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-surface3 border border-border2">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-teal/80 to-teal transition-all duration-500 ease-out shadow-[0_0_10px_rgba(20,184,166,0.4)]"
                        style={{ width: `${currentProgress}%` }}
                      />
                    </div>
                  </div>

                  {/* Live Status Message */}
                  <div className="mt-3 flex items-start gap-2 rounded border border-teal/20 bg-teal/5 p-2.5 text-[12px] leading-relaxed text-text2">
                    {jobStatus.status === 'completed' ? (
                      <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-teal" />
                    ) : jobStatus.status === 'failed' ? (
                      <AlertCircle size={16} className="mt-0.5 shrink-0 text-danger" />
                    ) : (
                      <LoaderCircle size={16} className="mt-0.5 shrink-0 animate-spin text-teal" />
                    )}
                    <span>{currentMessage}</span>
                  </div>
                </div>

                {/* Algorithm Stages Stepper */}
                <div className="rounded border border-border2 bg-surface p-4">
                  <p className="mb-3 font-mono text-[11px] font-bold uppercase tracking-[0.18em] text-text3">
                    Algorithm Pipeline Stages
                  </p>
                  <div className="space-y-2.5">
                    {ALGORITHM_STAGES.map(stage => {
                      let stageState: 'completed' | 'active' | 'pending' | 'failed' = 'pending'
                      if (jobStatus.status === 'completed') {
                        stageState = 'completed'
                      } else if (jobStatus.status === 'failed') {
                        stageState = currentProgress >= stage.minProgress ? 'completed' : 'failed'
                      } else if (stage.matchingStages.includes(jobStatus.stage) || (currentProgress >= stage.minProgress && currentProgress < stage.minProgress + 20)) {
                        stageState = 'active'
                      } else if (currentProgress > stage.minProgress) {
                        stageState = 'completed'
                      }

                      return (
                        <div
                          key={stage.id}
                          className={cn(
                            'flex items-start gap-3 rounded border p-2.5 transition-colors',
                            stageState === 'active'
                              ? 'border-teal/50 bg-teal/10'
                              : stageState === 'completed'
                                ? 'border-border2 bg-surface2/50'
                                : 'border-border/60 bg-surface/30 opacity-70'
                          )}
                        >
                          <div className="mt-0.5 shrink-0">
                            {stageState === 'completed' ? (
                              <CheckCircle2 size={16} className="text-teal" />
                            ) : stageState === 'active' ? (
                              <LoaderCircle size={16} className="animate-spin text-teal" />
                            ) : (
                              <span className="flex h-4 w-4 items-center justify-center rounded-full border border-border2 font-mono text-[10px] text-text3">
                                {stage.stepNumber}
                              </span>
                            )}
                          </div>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center justify-between gap-2">
                              <span className={cn(
                                'text-[12px] font-medium',
                                stageState === 'active' ? 'text-teal font-semibold' : stageState === 'completed' ? 'text-text1' : 'text-text3'
                              )}>
                                {stage.title}
                              </span>
                              <span className={cn(
                                'font-mono text-[10px] uppercase tracking-wider',
                                stageState === 'active' ? 'text-teal font-bold' : stageState === 'completed' ? 'text-text3' : 'text-text3/60'
                              )}>
                                {stageState === 'active' ? 'IN PROGRESS' : stageState === 'completed' ? 'DONE' : 'PENDING'}
                              </span>
                            </div>
                            <p className="mt-0.5 text-[11px] text-text3 leading-snug">
                              {stage.description}
                            </p>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Polling Indicator */}
                <div className="rounded border border-border2 bg-surface p-3">
                  <div className="flex items-center gap-2 text-[11px] font-mono text-text3">
                    {jobStatusQuery.isFetching && ACTIVE_STATUSES.includes(jobStatus.status) ? (
                      <>
                        <LoaderCircle size={12} className="animate-spin text-teal" />
                        <span>Live backend polling active (800ms updates)</span>
                      </>
                    ) : jobStatus.status === 'completed' ? (
                      <>
                        <CheckCircle2 size={12} className="text-teal" />
                        <span>Completed · Opening clinical result viewer</span>
                      </>
                    ) : jobStatus.status === 'failed' ? (
                      <>
                        <AlertCircle size={12} className="text-danger" />
                        <span>Analysis stopped with error</span>
                      </>
                    ) : (
                      <>
                        <Clock3 size={12} className="text-amber" />
                        <span>Waiting for next stage transition</span>
                      </>
                    )}
                  </div>
                </div>

                {jobFetchError && <ErrorBanner message="We couldn't retrieve the latest analysis status. Please refresh and try again." />}
              </div>
            )
          })()}
        </div>
      </div>

    </section>
  )
}
