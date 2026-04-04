import { useParams, useNavigate } from 'react-router-dom'
import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import TopNav from '@/components/layout/TopNav'
import StatBlock from '@/components/shared/StatBlock'
import VolumeChart from '@/components/scan/VolumeChart'
import ImagingHistory from '@/components/scan/ImagingHistory'
import AIInsightsPanel from '@/components/shared/AIInsightsPanel'
import MriWorkspace from '@/components/shared/MriWorkspace'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import { getPatient } from '@/api/patients'
import { getScans } from '@/api/scans'
import { getSummary } from '@/api/reports'
import { formatDate, formatVolume, calcVolumeDeltaPct } from '@/lib/utils'
import { ScanLine } from 'lucide-react'

export default function PatientDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()

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

  const patient = patientQuery.data
  const scans = useMemo(
    () => [...(scansQuery.data ?? [])].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()),
    [scansQuery.data]
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

  if (patientQuery.isError) {
    return (
      <div className="min-h-screen bg-bg">
        <TopNav />
        <div className="p-5">
          <ErrorBanner message="Patient not found." onRetry={() => navigate('/dashboard')} />
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      {/* Top bar */}
      <div className="h-[52px] bg-bg border-b border-border flex items-center gap-4 px-5 shrink-0">
        <button
          onClick={() => navigate('/dashboard')}
          className="text-[11px] font-mono text-text3 uppercase tracking-widest flex items-center gap-1.5 hover:text-text2 transition-colors"
        >
          ← Dashboard
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
        <div className="ml-auto shrink-0">
          <button className="border border-teal text-teal font-mono text-[11px] font-bold tracking-widest uppercase px-3.5 py-1.5 hover:bg-teal/5 transition-colors">
            ↓ Generate PDF Report
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Main column */}
        <div className="flex-1 overflow-y-auto p-5 flex flex-col gap-4 min-w-0">
          {/* Stats */}
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

          {/* Chart */}
          {scansQuery.isLoading ? (
            <div className="h-[164px] bg-surface border border-border animate-pulse" />
          ) : scans.length > 0 ? (
            <VolumeChart scans={scans} />
          ) : null}

          {/* History */}
          {scansQuery.isLoading ? (
            <div className="h-32 bg-surface border border-border animate-pulse" />
          ) : scans.length === 0 ? (
            <EmptyState
              icon={<ScanLine size={24} />}
              title="No imaging studies uploaded yet"
              description="Upload a DICOM study to begin longitudinal tracking."
            />
          ) : (
            <ImagingHistory scans={scans} />
          )}

          {/* AI Insights */}
          {summaryQuery.isLoading ? (
            <div className="h-28 bg-surface border border-border animate-pulse" />
          ) : summaryQuery.data ? (
            <AIInsightsPanel summary={summaryQuery.data} />
          ) : null}
        </div>

        {/* MRI sidebar — hidden on small screens */}
        {latestScan && (
          <div className="hidden lg:flex">
            <MriWorkspace scan={latestScan} />
          </div>
        )}
      </div>
    </div>
  )
}
