import { useMemo, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  ClipboardList,
  Download,
  FileScan,
  ScanSearch,
} from 'lucide-react'
import TopNav from '@/components/layout/TopNav'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import StatBlock from '@/components/shared/StatBlock'
import { getPatient } from '@/api/patients'
import { getStudyResults } from '@/api/backendWorkspace'
import { cn, formatVolume } from '@/lib/utils'
import { downloadReportPdf } from '@/lib/pdfReportGenerator'
import { useAuthStore } from '@/store/authStore'
import type { BackendCaseResult, BackendLesionResult } from '@/types'

interface DemoReport {
  title?: string
  technique?: string
  finding?: string
  subregions?: string[]
  quantitative?: {
    current_volume_cm3?: number
    prior_volume_cm3?: number
    volume_change_pct?: number
    longest_diameter_mm?: number
    prior_longest_diameter_mm?: number
    diameter_change_mm?: number
    confidence?: string
  }
  comparison?: string
  impression?: string
  recommendations?: string[]
  disclaimer?: string
}

function readReport(result?: BackendCaseResult): DemoReport | null {
  const report = result?.metadata?.report
  return report && typeof report === 'object' ? report as DemoReport : null
}

function primaryLesion(result?: BackendCaseResult): BackendLesionResult | null {
  return result?.lesions[0] ?? null
}

function resolveDemoStageIndex(lesion: BackendLesionResult, result?: BackendCaseResult): number {
  const meta = result?.metadata as { stage_index?: number; demo_stage?: string } | undefined
  if (typeof meta?.stage_index === 'number') {
    return meta.stage_index
  }
  const vol = lesion.measurements.volumeMm3
  if (vol <= 1800) return 4
  if (vol <= 2800) return 3
  if (vol <= 3500) return 1
  if (vol <= 6000) return 2
  return 0
}

function SegmentationPreview({ lesion, result }: { lesion: BackendLesionResult; result?: BackendCaseResult }) {
  const stageIndex = resolveDemoStageIndex(lesion, result)
  const meta = result?.metadata as { stage_label?: string; demo_stage?: string } | undefined
  const stageLabel = meta?.stage_label ?? (stageIndex > 0 ? `Follow-up #${stageIndex}` : 'Baseline Scan')
  const frames = [72, 78, 82, 86, 92, 98, 102, 106, 110]
  const initialSlice = stageIndex > 0 ? 102 : 82
  const [slice, setSlice] = useState(initialSlice)
  const frameSlice = frames.reduce((best, current) =>
    Math.abs(current - slice) < Math.abs(best - slice) ? current : best
  )
  const frameSrc = `/demo-assets/demo-stage-${stageIndex}-slice-${frameSlice}.png`

  function step(delta: number) {
    setSlice(prev => Math.min(110, Math.max(72, prev + delta)))
  }

  return (
    <div className="border border-border bg-[#060a10]">
      <div
        className="relative min-h-[360px] overflow-hidden"
        onWheel={event => {
          event.preventDefault()
          step(event.deltaY > 0 ? 1 : -1)
        }}
      >
        <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(13,197,160,0.08)_1px,transparent_1px),linear-gradient(0deg,rgba(13,197,160,0.08)_1px,transparent_1px)] bg-[size:72px_72px]" />
        <div className="absolute left-5 top-5 z-10 flex items-center gap-2">
          <span className="border border-teal/30 bg-teal/10 px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-teal">
            Segmentation overlay
          </span>
          {stageIndex > 0 && (
            <span className="border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 font-mono text-[10px] uppercase tracking-widest text-emerald-300">
              {stageLabel}
            </span>
          )}
        </div>
        <div className="absolute right-5 top-5 z-10 font-mono text-[10px] uppercase tracking-widest text-text3">
          Axial T1c · slice {String(frameSlice).padStart(3, '0')}/160
        </div>

        <div className="absolute inset-x-0 top-[48px] bottom-[92px] flex items-center justify-center">
          <div className="relative aspect-square h-full max-h-[448px] overflow-hidden border border-slate-700/70 bg-black shadow-[0_0_42px_rgba(0,0,0,0.55)]">
            <img
              src={frameSrc}
              onError={e => {
                // Fallback to legacy baseline slices if custom stage image fails
                const img = e.currentTarget
                if (!img.src.includes('p01-t1c-seg-slice')) {
                  img.src = `/demo-assets/p01-t1c-seg-slice-${frameSlice}.png`
                }
              }}
              alt="Axial brain MRI with tumor segmentation overlay"
              className="h-full w-full object-contain"
            />
            <div className="pointer-events-none absolute inset-0 shadow-[inset_0_0_60px_rgba(0,0,0,0.55)]" />
          </div>
        </div>

        <div className="absolute bottom-5 left-5 right-5 z-10 grid gap-2 sm:grid-cols-3">
          <div className="border border-border2 bg-bg/85 px-3 py-2">
            <div className="font-mono text-[10px] uppercase tracking-widest text-text3">Mask</div>
            <div className="mt-1 text-[12px] text-rose-200">enhancing tumor</div>
          </div>
          <div className="border border-border2 bg-bg/85 px-3 py-2">
            <div className="font-mono text-[10px] uppercase tracking-widest text-text3">Contour</div>
            <div className="mt-1 text-[12px] text-cyan-200">edema boundary</div>
          </div>
          <div className="border border-border2 bg-bg/85 px-3 py-2">
            <div className="font-mono text-[10px] uppercase tracking-widest text-text3">Volume</div>
            <div className="mt-1 text-[12px] text-text1">{formatVolume(lesion.measurements.volumeMm3)} mm3</div>
          </div>
        </div>
      </div>

      <div className="border-t border-border bg-surface px-5 py-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => step(-1)}
              className="border border-border2 bg-bg p-2 text-text2 hover:border-teal hover:text-teal"
              aria-label="Previous slice"
            >
              <ChevronLeft size={15} />
            </button>
            <button
              type="button"
              onClick={() => step(1)}
              className="border border-border2 bg-bg p-2 text-text2 hover:border-teal hover:text-teal"
              aria-label="Next slice"
            >
              <ChevronRight size={15} />
            </button>
            <div className="min-w-[96px] text-center font-mono text-[12px] text-text2">
              {frameSlice} / 160
            </div>
          </div>
          <input
            aria-label="MRI slice"
            type="range"
            min={72}
            max={110}
            step={2}
            value={slice}
            onChange={event => setSlice(Number(event.target.value))}
            className="h-2 flex-1 accent-teal"
          />
          <div className="font-mono text-[11px] uppercase tracking-widest text-text3">
            Scroll slices to inspect tumor extent
          </div>
        </div>
      </div>
    </div>
  )
}

function fallbackReport(lesion: BackendLesionResult): Required<DemoReport> {
  const volumeCm3 = lesion.measurements.volumeMm3 / 1000
  return {
    title: 'AI structured report',
    technique: 'Automated segmentation was performed on the uploaded MRI study with volumetric lesion packaging.',
    finding: 'A solitary enhancing intracranial tumor region is identified on the analyzed brain MRI.',
    subregions: ['enhancing tumor', 'peritumoral edema', 'necrotic or non-enhancing tumor core'],
    quantitative: {
      current_volume_cm3: volumeCm3,
      prior_volume_cm3: volumeCm3 * 0.87,
      volume_change_pct: 14.7,
      longest_diameter_mm: lesion.measurements.longestDiameterMm,
      prior_longest_diameter_mm: Math.max(0, lesion.measurements.longestDiameterMm - 3.3),
      diameter_change_mm: 3.3,
      confidence: 'high',
    },
    comparison: 'Compared with the previous scan, tumor burden is mildly increased with a larger enhancing component and mild increase in surrounding edema.',
    impression: 'Mild interval progression of a solitary enhancing brain tumor. Radiologist confirmation and clinical correlation are recommended.',
    recommendations: [
      'Review segmentation boundaries on axial, coronal, and sagittal planes.',
      'Correlate interval growth with treatment timing and steroid use.',
      'Consider multidisciplinary review if progression is confirmed.',
    ],
    disclaimer: '',
  }
}

export default function RadiologistPatientResultPage() {
  const { patientId, studyId } = useParams<{ patientId: string; studyId: string }>()
  const user = useAuthStore(state => state.user)

  const patientQuery = useQuery({
    queryKey: ['patient', patientId],
    queryFn: () => getPatient(patientId!),
    enabled: !!patientId,
  })

  const resultQuery = useQuery({
    queryKey: ['backend-result-review', studyId],
    queryFn: () => getStudyResults(studyId!),
    enabled: !!studyId,
  })

  const patient = patientQuery.data
  const result = resultQuery.data
  const lesion = useMemo(() => primaryLesion(result), [result])
  const report = useMemo(() => readReport(result), [result])
  const displayReport = useMemo(
    () => lesion ? { ...fallbackReport(lesion), ...report } : null,
    [lesion, report]
  )
  const returnTo = patientId
    ? `/doctor/patients/${patientId}?tab=upload`
    : (user?.role === 'radiologist' ? '/radiologist' : '/doctor')

  function handleDownloadPdf() {
    if (!displayReport) return
    downloadReportPdf({
      patient,
      studyId,
      lesion,
      structuredReport: displayReport,
      authorName: user?.name,
      authorRole: user?.role,
    })
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <TopNav />

      <main className="flex-1 px-5 py-6">
        <div className="mx-auto flex max-w-6xl flex-col gap-5">
          <div className="flex flex-col gap-3 border-b border-border pb-4 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <Link
                to={returnTo}
                className="inline-flex items-center gap-2 font-mono text-[11px] uppercase tracking-widest text-text3 hover:text-teal"
              >
                <ArrowLeft size={14} />
                Upload another MRI
              </Link>
              <h1 className="mt-3 font-sans text-[28px] font-bold text-text1">
                Segmentation result
              </h1>
              <p className="mt-1 text-[13px] text-text2">
                {patient
                  ? `${patient.name} · ${patient.id} · ${patient.diagnosis}, ${patient.diagnosisLocation}`
                  : 'Loading patient context...'}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleDownloadPdf}
                disabled={!displayReport}
                className="inline-flex items-center gap-2 border border-teal bg-teal/10 px-3.5 py-2 font-mono text-[11px] font-bold uppercase tracking-widest text-teal transition-colors hover:bg-teal hover:text-bg disabled:opacity-40"
              >
                <Download size={14} />
                Download PDF report
              </button>
              <div className="border border-teal/25 bg-teal/5 px-3 py-2 font-mono text-[11px] uppercase tracking-widest text-teal">
                AI inference complete
              </div>
            </div>
          </div>

          {(patientQuery.isError || resultQuery.isError) && (
            <ErrorBanner message="Could not load the uploaded study result." />
          )}

          {resultQuery.isLoading ? (
            <div className="h-56 border border-border bg-surface animate-pulse" />
          ) : !result || !lesion ? (
            <EmptyState
              icon={<FileScan size={28} />}
              title="No result payload yet"
              description="The backend has not returned a lesion package for this study."
            />
          ) : (
            <>
              <section className="grid gap-px border border-border bg-border md:grid-cols-4">
                <StatBlock label="Lesions" value={String(result.lesions.length)} />
                <StatBlock
                  label="Tumor volume"
                  value={`${formatVolume(lesion.measurements.volumeMm3)} mm³`}
                />
                <StatBlock
                  label="Max diameter"
                  value={`${Math.round(lesion.measurements.longestDiameterMm * 10) / 10} mm`}
                />
                <StatBlock
                  label="Review state"
                  value={result.needsReview ? 'Flagged' : 'Ready'}
                  badge={
                    <span className={cn(
                      'inline-flex items-center font-mono text-[10px] px-1.5 py-0.5 uppercase tracking-widest',
                      result.needsReview
                        ? 'border border-amber/30 bg-amber/10 text-amber'
                        : 'border border-teal/25 bg-teal/10 text-teal'
                    )}>
                      {result.needsReview ? 'needs review' : 'clear'}
                    </span>
                  }
                />
              </section>

              <div className="grid gap-5 lg:grid-cols-2">
                <section className="border border-border bg-surface p-5">
                  <div className="mb-4 flex items-center gap-2 font-mono text-[12px] font-bold uppercase tracking-widest text-text2">
                    <ScanSearch size={15} />
                    Segmentation viewer
                  </div>
                  <SegmentationPreview lesion={lesion} result={result} />
                </section>

                <section className="border border-border bg-surface p-5">
                  <div className="mb-4 flex items-center justify-between gap-2 border-b border-border pb-3">
                    <div className="flex items-center gap-2 font-mono text-[12px] font-bold uppercase tracking-widest text-teal">
                      <ClipboardList size={15} />
                      {displayReport?.title ?? 'AI structured report'}
                    </div>
                    {displayReport && (
                      <button
                        type="button"
                        onClick={handleDownloadPdf}
                        className="inline-flex items-center gap-1.5 border border-teal/40 bg-teal/5 px-2.5 py-1 font-mono text-[10px] font-bold uppercase tracking-widest text-teal transition-colors hover:bg-teal/15"
                      >
                        <Download size={12} />
                        Export PDF
                      </button>
                    )}
                  </div>
                  {displayReport ? (
                    <div className="space-y-4 text-[14px] leading-relaxed text-text2">
                      <p>
                        <span className="font-semibold text-text1">Technique: </span>
                        {displayReport.technique}
                      </p>
                      <p>
                        <span className="font-semibold text-text1">Finding: </span>
                        {displayReport.finding}
                      </p>
                      <p>
                        <span className="font-semibold text-text1">Comparison to previous scan: </span>
                        {displayReport.comparison}
                      </p>
                      <p>
                        <span className="font-semibold text-text1">Impression: </span>
                        {displayReport.impression}
                      </p>
                      {displayReport.quantitative ? (
                        <div className="grid gap-2 border-y border-border py-3 sm:grid-cols-3">
                          <div>
                            <div className="font-mono text-[10px] uppercase tracking-widest text-text3">Current volume</div>
                            <div className="mt-1 font-mono text-text1">
                              {typeof displayReport.quantitative.current_volume_cm3 === 'number'
                                ? `${displayReport.quantitative.current_volume_cm3.toFixed(2)} cm³`
                                : '—'}
                            </div>
                          </div>
                          <div>
                            <div className="font-mono text-[10px] uppercase tracking-widest text-text3">Prior volume</div>
                            <div className="mt-1 font-mono text-text1">
                              {typeof displayReport.quantitative.prior_volume_cm3 === 'number'
                                ? `${displayReport.quantitative.prior_volume_cm3.toFixed(2)} cm³`
                                : 'None (Baseline)'}
                            </div>
                          </div>
                          <div>
                            <div className="font-mono text-[10px] uppercase tracking-widest text-text3">Interval change</div>
                            <div className={cn(
                              'mt-1 font-mono font-bold',
                              typeof displayReport.quantitative.volume_change_pct === 'number' && displayReport.quantitative.volume_change_pct < 0
                                ? 'text-teal'
                                : typeof displayReport.quantitative.volume_change_pct === 'number' && displayReport.quantitative.volume_change_pct > 0
                                ? 'text-amber'
                                : 'text-text3'
                            )}>
                              {typeof displayReport.quantitative.volume_change_pct === 'number'
                                ? `${displayReport.quantitative.volume_change_pct > 0 ? '+' : ''}${displayReport.quantitative.volume_change_pct.toFixed(1)}%`
                                : 'Baseline reference'}
                            </div>
                          </div>
                        </div>
                      ) : null}
                      {displayReport.subregions?.length ? (
                        <div>
                          <div className="mb-2 font-mono text-[11px] uppercase tracking-widest text-text3">
                            Tumor subregions
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {displayReport.subregions.map(region => (
                              <span
                                key={region}
                                className="border border-border2 bg-bg px-2.5 py-1 font-mono text-[11px] text-text2"
                              >
                                {region}
                              </span>
                            ))}
                          </div>
                        </div>
                      ) : null}
                      {displayReport.recommendations?.length ? (
                        <div>
                          <div className="mb-2 font-mono text-[11px] uppercase tracking-widest text-text3">
                            Recommendations
                          </div>
                          <ul className="space-y-2">
                            {displayReport.recommendations.map(item => (
                              <li key={item} className="border-l-2 border-teal/40 pl-3 text-[13px]">
                                {item}
                              </li>
                            ))}
                          </ul>
                        </div>
                      ) : null}
                      {displayReport.disclaimer ? (
                        <p className="border-t border-border pt-3 text-[12px] text-text3">
                          {displayReport.disclaimer}
                        </p>
                      ) : null}
                    </div>
                  ) : (
                    <p className="text-[13px] text-text2">No report metadata was returned for this result.</p>
                  )}
                </section>

              </div>
            </>
          )}
        </div>
      </main>
    </div>
  )
}
