import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import StatBlock from '@/components/shared/StatBlock'
import VolumeChart from '@/components/scan/VolumeChart'
import ImagingHistory from '@/components/scan/ImagingHistory'
import AIInsightsPanel from '@/components/shared/AIInsightsPanel'
import MriWorkspace from '@/components/shared/MriWorkspace'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import BackendOperatorWorkspace from '@/components/dashboard/BackendOperatorWorkspace'
import LongitudinalComparisonPanel from '@/components/dashboard/LongitudinalComparisonPanel'
import { getPatient } from '@/api/patients'
import { getScans } from '@/api/scans'
import { generateReport, getSummary, listReports } from '@/api/reports'
import { formatDate, formatVolume, calcVolumeDeltaPct, cn } from '@/lib/utils'
import { ScanLine } from 'lucide-react'

type Tab = 'scans' | 'longitudinal' | 'upload' | 'reports'

function PatientTabButton({
  active,
  label,
  onClick,
}: {
  active: boolean
  label: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'px-4 py-2 font-mono text-[11px] font-bold uppercase tracking-widest border-b-2 transition-colors',
        active ? 'border-teal text-text1' : 'border-transparent text-text3 hover:text-text2'
      )}
    >
      {label}
    </button>
  )
}

export default function DoctorPatientDashboardPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const queryClient = useQueryClient()

  const tabParam = searchParams.get('tab') as Tab | null
  const [tab, setTabState] = useState<Tab>(() => {
    if (tabParam && ['scans', 'longitudinal', 'upload', 'reports'].includes(tabParam)) {
      return tabParam
    }
    return 'scans'
  })

  useEffect(() => {
    if (tabParam && ['scans', 'longitudinal', 'upload', 'reports'].includes(tabParam)) {
      setTabState(tabParam)
    }
  }, [tabParam])

  function setTab(newTab: Tab) {
    setTabState(newTab)
    setSearchParams(prev => {
      const next = new URLSearchParams(prev)
      next.set('tab', newTab)
      return next
    }, { replace: true })
  }

  const [selectedScanId, setSelectedScanId] = useState<string | null>(null)

  const patientQuery = useQuery({
    queryKey: ['patient', id],
    queryFn: () => getPatient(id!),
    enabled: !!id,
  })

  const scansQuery = useQuery({
    queryKey: ['scans', id],
    queryFn: () => getScans(id!),
    enabled: !!id,
  })

  const summaryQuery = useQuery({
    queryKey: ['summary', id],
    queryFn: () => getSummary(id!),
    enabled: !!id,
  })

  const reportsQuery = useQuery({
    queryKey: ['reports', id],
    queryFn: () => listReports(id!),
    enabled: !!id && tab === 'reports',
  })

  const generateMutation = useMutation({
    mutationFn: () => generateReport(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['reports', id] })
    },
  })

  const patient = patientQuery.data
  const scans = useMemo(
    () => [...(scansQuery.data ?? [])].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()),
    [scansQuery.data]
  )

  useEffect(() => {
    if (scans.length === 0) {
      setSelectedScanId(null)
      return
    }
    setSelectedScanId(prev => {
      if (prev && scans.some(s => s.id === prev)) return prev
      return scans[scans.length - 1].id
    })
  }, [scans])

  const selectedScan = useMemo(
    () => scans.find(s => s.id === selectedScanId) ?? scans[scans.length - 1],
    [scans, selectedScanId]
  )

  const latestScan = scans[scans.length - 1]
  const previousScan = scans[scans.length - 2]

  const volumeDelta = latestScan && previousScan
    ? calcVolumeDeltaPct(latestScan.volumeMm3, previousScan.volumeMm3)
    : null

  const diameterDelta = latestScan && previousScan
    ? Math.round((latestScan.maxDiameterMm - previousScan.maxDiameterMm) * 10) / 10
    : null

  const allAnnotated = scans.length > 0 && scans.every(s => s.isAnnotated)

  const scopeCopy =
    patient?.linkedStudyIds?.length
      ? undefined
      : 'Studies here come from the live demo backend (all completed analyses). Complete radiologist uploads first so P01 labels appear; roster patients are not linked to backend UUIDs until that feature ships.'

  if (patientQuery.isError) {
    return (
      <div className="min-h-screen bg-bg">
        <div className="h-[52px] bg-bg border-b border-border px-5 flex items-center">
          <button
            type="button"
            onClick={() => navigate('/doctor')}
            className="text-[11px] font-mono text-text3 uppercase tracking-widest hover:text-text2"
          >
            ← Patient roster
          </button>
        </div>
        <div className="p-5">
          <ErrorBanner message="Patient not found." onRetry={() => navigate('/doctor')} />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <div className="h-[52px] bg-bg border-b border-border flex items-center gap-4 px-5 shrink-0">
        <button
          type="button"
          onClick={() => navigate('/doctor')}
          className="text-[11px] font-mono text-text3 uppercase tracking-widest flex items-center gap-1.5 hover:text-text2 transition-colors"
        >
          ← Patient roster
        </button>
        <span className="text-border2">|</span>
        {patient ? (
          <div className="flex items-baseline gap-2 min-w-0">
            <span className="font-sans font-bold text-[18px] text-text1 truncate">{patient.name}</span>
            <span className="font-mono text-[11px] text-teal">{patient.id}</span>
            <span className="text-[12px] text-text2 truncate hidden sm:block">
              · {patient.diagnosis}, {patient.diagnosisLocation} · DOB {formatDate(patient.dob)}
            </span>
          </div>
        ) : (
          <div className="h-4 w-48 bg-surface3 animate-pulse" />
        )}
      </div>

      <div className="border-b border-border px-5 flex gap-1 bg-surface/40">
        <PatientTabButton active={tab === 'scans'} label="Scans & viewer" onClick={() => setTab('scans')} />
        <PatientTabButton active={tab === 'longitudinal'} label="Longitudinal" onClick={() => setTab('longitudinal')} />
        <PatientTabButton active={tab === 'upload'} label="Upload MRI — pipeline" onClick={() => setTab('upload')} />
        <PatientTabButton active={tab === 'reports'} label="Reports" onClick={() => setTab('reports')} />
      </div>

      {tab === 'scans' && (
        <div className="flex flex-1 overflow-hidden">
          <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-w-0">
            <div className="grid grid-cols-4 gap-px bg-border border border-border">
              <StatBlock
                label="Total Scans"
                value={scans.length.toString()}
                badge={
                  <span className="inline-flex items-center font-mono text-[10px] bg-surface2 text-text2 px-1.5 py-0.5">
                    {scans.length > 1 ? `${scans.length} studies` : '1 study'}
                  </span>
                }
              />
              <StatBlock
                label="Current Volume"
                value={latestScan ? `${formatVolume(latestScan.volumeMm3)} mm³` : '—'}
                delta={volumeDelta}
                deltaUnit="%"
              />
              <StatBlock
                label="Max Diameter"
                value={latestScan ? `${latestScan.maxDiameterMm} mm` : '—'}
                delta={diameterDelta}
                deltaUnit=" mm"
              />
              <StatBlock
                label="Annotated"
                value=""
                badge={
                  <span className={`inline-flex items-center font-mono text-[11px] font-bold px-2 py-1 ${
                    allAnnotated
                      ? 'bg-teal/10 border border-teal/25 text-teal'
                      : 'bg-surface3 border border-border2 text-amber'
                  }`}>
                    {allAnnotated ? '✓ ALL SCANS' : '○ PARTIAL'}
                  </span>
                }
              />
            </div>

            {scansQuery.isLoading ? (
              <div className="h-[164px] bg-surface border border-border animate-pulse" />
            ) : scans.length > 0 ? (
              <VolumeChart scans={scans} />
            ) : null}

            <p className="text-[11px] font-mono text-text3 uppercase tracking-widest">
              Slice viewer (multi-planar); full volumetric 3D rendering can plug in here later.
            </p>

            {scansQuery.isLoading ? (
              <div className="h-32 bg-surface border border-border animate-pulse" />
            ) : scans.length === 0 ? (
              <EmptyState
                icon={<ScanLine size={24} />}
                title="No imaging studies uploaded yet"
                description="Ask radiology to upload a scan — it will appear here after ingestion."
              />
            ) : (
              <ImagingHistory
                scans={scans}
                selectedScanId={selectedScan?.id}
                onSelectScan={s => setSelectedScanId(s.id)}
              />
            )}

            {summaryQuery.isLoading ? (
              <div className="h-28 bg-surface border border-border animate-pulse" />
            ) : summaryQuery.data ? (
              <AIInsightsPanel summary={summaryQuery.data} />
            ) : null}
          </div>

          {selectedScan && (
            <div className="hidden lg:flex">
              <MriWorkspace scan={selectedScan} />
            </div>
          )}
        </div>
      )}

      {tab === 'longitudinal' && (
        <div className="flex-1 overflow-y-auto p-5">
          <LongitudinalComparisonPanel
            restrictToStudyIds={patient?.linkedStudyIds}
            scopeNote={
              patient?.linkedStudyIds?.length
                ? undefined
                : 'Showing all demo-backend studies. Configure linkedStudyIds on the mock patient to narrow comparisons.'
            }
          />
        </div>
      )}
      {tab === 'upload' && (
        <div className="flex-1 overflow-y-auto p-5">
          <BackendOperatorWorkspace
            headingEyebrow="RADIOLOGIST WORKSPACE"
            headingTitle="Upload MRI — segmentation pipeline"
            headingDescription="After ingestion completes, we attempt an automatic longitudinal comparison between the earliest and latest backend studies that have stored results (optionally filtered by this patient's linkedStudyIds when configured)."
            prefilledSourceLabel={patient ? `${patient.id} · ${patient.name}` : ''}
            onJobReachedTerminal={() => {
              queryClient.invalidateQueries({ queryKey: ['backend-studies'] })
              queryClient.invalidateQueries({ queryKey: ['scans', id] })
            }}
          />
        </div>
      )}

      {tab === 'reports' && (
        <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 max-w-3xl">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => generateMutation.mutate()}
              disabled={generateMutation.isPending || !id}
              className="border border-teal text-teal font-mono text-[11px] font-bold tracking-widest uppercase px-3.5 py-1.5 hover:bg-teal/5 transition-colors disabled:opacity-40"
            >
              {generateMutation.isPending ? 'Generating…' : 'Generate report'}
            </button>
            {generateMutation.isError && (
              <span className="text-danger text-[12px] font-mono">Could not generate report.</span>
            )}
          </div>

          <div className="text-[12px] font-mono font-bold tracking-widest uppercase text-text2">
            Previous reports (mock storage)
          </div>

          {reportsQuery.isLoading ? (
            <div className="h-24 bg-surface border border-border animate-pulse" />
          ) : (reportsQuery.data?.length ?? 0) === 0 ? (
            <p className="text-[13px] text-text2">No saved reports yet — generate one above.</p>
          ) : (
            <ul className="space-y-2">
              {reportsQuery.data!.map(r => (
                <li key={r.id} className="border border-border bg-surface p-4">
                  <div className="font-mono text-[11px] text-teal">{r.title}</div>
                  <div className="text-[11px] font-mono text-text3 mt-1">
                    {new Date(r.generatedAt).toLocaleString()}
                  </div>
                  <p className="text-[13px] text-text2 mt-2 leading-relaxed">{r.summarySnippet}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

