import { useMemo, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Activity, ArrowRight, GitCompareArrows, LoaderCircle, RefreshCw } from 'lucide-react'
import {
  BackendApiError,
  listStudies,
  submitComparison,
} from '@/api/backendWorkspace'
import EmptyState from '@/components/shared/EmptyState'
import ErrorBanner from '@/components/shared/ErrorBanner'
import StatBlock from '@/components/shared/StatBlock'
import { cn } from '@/lib/utils'
import type {
  BackendComparisonResponse,
  BackendStudyListItem,
} from '@/types'

function studyDisplayName(study: BackendStudyListItem): string {
  const label = study.sourceLabel?.trim() || `Study ${study.studyId.slice(0, 8)}`
  const date = study.acquiredAt ? ` · ${study.acquiredAt}` : ''
  return `${label}${date}`
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2)
}

function fmtNumber(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) return '—'
  return value.toFixed(digits)
}

function pctChangeBadgeClass(pct: number): string {
  if (Math.abs(pct) < 5) return 'border-border2 bg-surface2 text-text2'
  return pct > 0
    ? 'border-danger/30 bg-danger/10 text-danger'
    : 'border-teal/25 bg-teal/10 text-teal'
}

function pctChangeLabel(pct: number): string {
  if (Math.abs(pct) <= 5) return 'stable'
  if (pct >= 25) return 'progressive'
  if (pct <= -25) return 'response'
  return 'minor change'
}

interface LongitudinalComparisonPanelProps {
  /** When set, only these backend study IDs appear in the selectors */
  restrictToStudyIds?: string[]
  scopeNote?: string
}

export default function LongitudinalComparisonPanel({
  restrictToStudyIds,
  scopeNote,
}: LongitudinalComparisonPanelProps = {}) {
  const [baselineId, setBaselineId] = useState<string>('')
  const [followupId, setFollowupId] = useState<string>('')
  const [submitted, setSubmitted] = useState<BackendComparisonResponse | null>(null)

  const studiesQuery = useQuery({
    queryKey: ['backend-studies'],
    queryFn: listStudies,
    refetchInterval: 5000,
  })

  const eligibleStudies = useMemo(() => {
    let list = (studiesQuery.data ?? []).filter(s => s.hasResults)
    if (restrictToStudyIds?.length) {
      list = list.filter(s => restrictToStudyIds.includes(s.studyId))
    }
    return list
  }, [studiesQuery.data, restrictToStudyIds])

  const comparisonMutation = useMutation({
    mutationFn: ({ baseline, followup }: { baseline: string; followup: string }) =>
      submitComparison({ baselineStudyId: baseline, followupStudyId: followup }),
    onSuccess: result => setSubmitted(result),
  })

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!baselineId || !followupId) return
    if (baselineId === followupId) return
    comparisonMutation.mutate({ baseline: baselineId, followup: followupId })
  }

  const studiesError =
    studiesQuery.error instanceof BackendApiError ? studiesQuery.error : null
  const compareError =
    comparisonMutation.error instanceof BackendApiError ? comparisonMutation.error : null

  const canSubmit =
    !!baselineId && !!followupId && baselineId !== followupId && !comparisonMutation.isPending

  return (
    <section className="border border-border bg-surface">
      <div className="border-b border-border px-5 py-4">
        <div className="flex flex-col gap-2 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-teal">Longitudinal Comparison</p>
            <h2 className="mt-1 text-[24px] font-sans font-bold text-text1">
              Compare two scans, see tumor change
            </h2>
            <p className="mt-2 max-w-3xl text-[13px] leading-relaxed text-text2">
              Pick a baseline study and a follow-up study. The backend runs registration, computes volume change,
              Dice / HD95 overlap, and RECIST diameters using the lightweight base-models pipeline.
            </p>
            {scopeNote && (
              <p className="mt-3 max-w-3xl rounded border border-border2 bg-bg px-3 py-2 text-[12px] leading-relaxed text-text2">
                {scopeNote}
              </p>
            )}
          </div>
          <div className="rounded border border-border2 bg-bg px-3 py-2 font-mono text-[11px] text-text3">
            POST <span className="text-text1">/api/v1/jobs/longitudinal-comparison</span>
            <br />
            GET <span className="text-text1">/api/v1/results/studies</span>
          </div>
        </div>
      </div>

      <div className="grid gap-px border-b border-border bg-border lg:grid-cols-[1.15fr,0.85fr]">
        <div className="bg-bg p-5">
          <form className="space-y-4" onSubmit={handleSubmit}>
            <div className="flex items-center gap-2 text-[12px] font-mono uppercase tracking-[0.18em] text-text3">
              <GitCompareArrows size={14} />
              Choose timepoints
            </div>

            {studiesError && <ErrorBanner message={studiesError.message} />}

            <div className="space-y-3">
              <div>
                <label htmlFor="baseline-study" className="mb-2 block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                  Baseline Study
                </label>
                <select
                  id="baseline-study"
                  value={baselineId}
                  onChange={event => setBaselineId(event.target.value)}
                  className="w-full border border-border2 bg-surface px-3.5 py-[10px] text-[14px] text-text1 focus:border-teal focus:outline-none"
                >
                  <option value="">Select baseline…</option>
                  {eligibleStudies.map(study => (
                    <option key={study.studyId} value={study.studyId}>
                      {studyDisplayName(study)}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center justify-center text-text3">
                <ArrowRight size={18} />
              </div>

              <div>
                <label htmlFor="followup-study" className="mb-2 block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                  Follow-up Study
                </label>
                <select
                  id="followup-study"
                  value={followupId}
                  onChange={event => setFollowupId(event.target.value)}
                  className="w-full border border-border2 bg-surface px-3.5 py-[10px] text-[14px] text-text1 focus:border-teal focus:outline-none"
                >
                  <option value="">Select follow-up…</option>
                  {eligibleStudies
                    .filter(study => study.studyId !== baselineId)
                    .map(study => (
                      <option key={study.studyId} value={study.studyId}>
                        {studyDisplayName(study)}
                      </option>
                    ))}
                </select>
              </div>
            </div>

            {compareError && <ErrorBanner message={compareError.message} />}

            <div className="flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={!canSubmit}
                className="bg-teal px-4 py-2.5 font-mono text-[12px] font-bold uppercase tracking-[0.18em] text-black transition-colors hover:bg-teal/90 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {comparisonMutation.isPending ? 'Running…' : 'Run Comparison'}
              </button>
              <button
                type="button"
                onClick={() => studiesQuery.refetch()}
                className="inline-flex items-center gap-1.5 border border-border2 bg-surface px-3 py-2 font-mono text-[11px] font-bold uppercase tracking-[0.18em] text-text2 hover:border-teal hover:text-teal"
              >
                <RefreshCw size={12} />
                Refresh studies
              </button>
              <span className="text-[12px] text-text3">
                {eligibleStudies.length} completed stud{eligibleStudies.length === 1 ? 'y' : 'ies'} available.
              </span>
            </div>
          </form>
        </div>

        <div className="bg-bg p-5">
          <div className="mb-4 flex items-center gap-2 text-[12px] font-mono uppercase tracking-[0.18em] text-text3">
            <Activity size={14} />
            Status
          </div>

          {comparisonMutation.isPending ? (
            <div className="rounded border border-border2 bg-surface p-4 text-[13px] text-text2">
              <span className="inline-flex items-center gap-2">
                <LoaderCircle size={14} className="animate-spin text-teal" />
                Running registration and metrics. This usually takes a few seconds with provided masks.
              </span>
            </div>
          ) : submitted ? (
            <div className="rounded border border-border2 bg-surface p-4 text-[13px] text-text2">
              <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Comparison ID</p>
              <p className="mt-1 break-all font-mono text-text1">{submitted.comparisonId}</p>
              <p className="mt-3 text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Output path</p>
              <p className="mt-1 break-all font-mono text-text1">derived/{submitted.outputRelativePath}</p>
            </div>
          ) : (
            <EmptyState
              icon={<GitCompareArrows size={22} />}
              title="No comparison submitted yet"
              description="Choose two studies above and run a comparison to see growth metrics."
              className="border border-dashed border-border2 bg-surface"
            />
          )}
        </div>
      </div>

      <div className="bg-bg p-5">
        <div className="mb-4 flex items-center gap-2 text-[12px] font-mono uppercase tracking-[0.18em] text-text3">
          <Activity size={14} />
          Comparison metrics
        </div>

        {!submitted ? (
          <EmptyState
            icon={<GitCompareArrows size={24} />}
            title="Metrics will appear after a comparison runs"
            description="Volumes, percent change, RECIST, Dice / HD95, and growth rate are reported here once the backend returns."
            className="min-h-[220px] border border-dashed border-border2 bg-surface"
          />
        ) : (
          <div className="space-y-5">
            <div className="grid gap-px border border-border bg-border lg:grid-cols-4">
              <StatBlock
                label="Baseline Volume"
                value={`${fmtNumber(submitted.metrics.volumeACm3, 2)} cm³`}
              />
              <StatBlock
                label="Follow-up Volume"
                value={`${fmtNumber(submitted.metrics.volumeBCm3, 2)} cm³`}
              />
              <StatBlock
                label="Δ Volume"
                value={`${submitted.metrics.deltaCm3 > 0 ? '+' : ''}${fmtNumber(
                  submitted.metrics.deltaCm3,
                  2
                )} cm³`}
                badge={
                  <span
                    className={cn(
                      'inline-flex items-center font-mono text-[10px] px-1.5 py-0.5 uppercase tracking-widest border',
                      submitted.metrics.deltaCm3 > 0
                        ? 'border-danger/30 bg-danger/10 text-danger'
                        : 'border-teal/25 bg-teal/10 text-teal'
                    )}
                  >
                    {submitted.metrics.deltaCm3 > 0 ? 'growth' : 'shrinkage'}
                  </span>
                }
              />
              <StatBlock
                label="% Change"
                value={`${submitted.metrics.pctChange > 0 ? '+' : ''}${fmtNumber(
                  submitted.metrics.pctChange,
                  1
                )}%`}
                badge={
                  <span
                    className={cn(
                      'inline-flex items-center font-mono text-[10px] px-1.5 py-0.5 uppercase tracking-widest border',
                      pctChangeBadgeClass(submitted.metrics.pctChange)
                    )}
                  >
                    {pctChangeLabel(submitted.metrics.pctChange)}
                  </span>
                }
              />
            </div>

            <div className="grid gap-px border border-border bg-border lg:grid-cols-4">
              <StatBlock
                label="RECIST A"
                value={`${fmtNumber(submitted.metrics.recistAMm, 1)} mm`}
              />
              <StatBlock
                label="RECIST B"
                value={`${fmtNumber(submitted.metrics.recistBMm, 1)} mm`}
              />
              <StatBlock
                label="RECIST Ratio"
                value={fmtNumber(submitted.metrics.recistRatio, 3)}
              />
              <StatBlock
                label="Growth Rate"
                value={`${fmtNumber(submitted.metrics.growthRateCm3PerDay, 4)} cm³/day`}
              />
            </div>

            <div className="grid gap-px border border-border bg-border lg:grid-cols-4">
              <StatBlock
                label="Dice Overlap"
                value={fmtNumber(submitted.metrics.diceOverlap, 3)}
              />
              <StatBlock
                label="HD95"
                value={`${fmtNumber(submitted.metrics.hd95Mm, 1)} mm`}
              />
              <StatBlock
                label="Reg. NCC"
                value={fmtNumber(submitted.metrics.registrationNcc, 3)}
              />
              <StatBlock
                label="Method"
                value={(submitted.metrics.method ?? '—').toString()}
              />
            </div>

            {submitted.interpretation && (
              <div className="rounded border border-teal/30 bg-teal/10 p-4">
                <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-teal">
                  Interpretation
                </p>
                <p className="mt-2 text-[16px] font-semibold text-text1">{submitted.interpretation}</p>
              </div>
            )}

            {submitted.notes.length > 0 && (
              <div className="rounded border border-border2 bg-surface p-4">
                <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-text3">Notes</p>
                <ul className="mt-2 space-y-1 text-[13px] text-text2">
                  {submitted.notes.map(note => (
                    <li key={note}>{note}</li>
                  ))}
                </ul>
              </div>
            )}

            <details className="rounded border border-border2 bg-surface p-4">
              <summary className="cursor-pointer text-[11px] font-mono uppercase tracking-[0.18em] text-text3">
                Raw comparison payload
              </summary>
              <pre className="mt-3 overflow-x-auto whitespace-pre-wrap break-words text-[11px] text-text2">
                {prettyJson(submitted)}
              </pre>
            </details>
          </div>
        )}
      </div>
    </section>
  )
}
