import { useQuery } from '@tanstack/react-query'
import TopNav from '@/components/layout/TopNav'
import StatBlock from '@/components/shared/StatBlock'
import AIInsightsPanel from '@/components/shared/AIInsightsPanel'
import EmptyState from '@/components/shared/EmptyState'
import ImagingHistory from '@/components/scan/ImagingHistory'
import ErrorBanner from '@/components/shared/ErrorBanner'
import { useAuthStore } from '@/store/authStore'
import { getPatient } from '@/api/patients'
import { getScans } from '@/api/scans'
import { getSummary, listReports } from '@/api/reports'
import { formatDate } from '@/lib/utils'
import { Activity, FileText } from 'lucide-react'

const DEFAULT_RECOMMENDATIONS = [
  'Bring imaging summaries to your next oncology appointment.',
  'Do not change medications without speaking to your physician.',
]

export default function PatientPortalPage() {
  const recordId = useAuthStore(s => s.user?.patientRecordId ?? s.user?.id)

  const patientQuery = useQuery({
    queryKey: ['patient', recordId],
    queryFn: () => getPatient(recordId!),
    enabled: !!recordId,
  })

  const scansQuery = useQuery({
    queryKey: ['scans', recordId],
    queryFn: () => getScans(recordId!),
    enabled: !!recordId,
  })

  const summaryQuery = useQuery({
    queryKey: ['summary', recordId],
    queryFn: () => getSummary(recordId!),
    enabled: !!recordId,
  })

  const reportsQuery = useQuery({
    queryKey: ['reports', recordId],
    queryFn: () => listReports(recordId!),
    enabled: !!recordId,
  })

  const patient = patientQuery.data
  const scans = scansQuery.data ?? []
  const recommendations = summaryQuery.data?.recommendations ?? DEFAULT_RECOMMENDATIONS

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <TopNav />

      <main className="flex-1 px-5 py-6 flex flex-col gap-6 max-w-4xl mx-auto w-full">
        {patientQuery.isError && (
          <ErrorBanner message="We could not load your chart; check your patient ID." />
        )}

        {patient && (
          <>
            <div>
              <h1 className="font-sans font-bold text-[26px] text-text1">{patient.name}</h1>
              <p className="text-[13px] text-text2 mt-1 font-mono">
                {patient.id} · DOB {formatDate(patient.dob)} · {patient.diagnosis}
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-border border border-border">
              <StatBlock label="Care status" value={patient.status} />
              <StatBlock label="Scans on file" value={String(scans.length)} />
              <StatBlock label="Last study" value={patient.lastScanDate ? formatDate(patient.lastScanDate) : '—'} />
              <StatBlock label="Site" value={patient.diagnosisLocation} />
            </div>

            <section className="border border-border bg-surface p-5">
              <div className="flex items-center gap-2 text-[12px] font-mono font-bold uppercase tracking-widest text-text2 mb-4">
                <Activity size={16} />
                Imaging timeline (read-only)
              </div>
              {scans.length === 0 ? (
                <p className="text-[13px] text-text2">No scans are visible in your portal yet.</p>
              ) : (
                <ImagingHistory scans={scans} />
              )}
            </section>

            {summaryQuery.data && (
              <AIInsightsPanel summary={summaryQuery.data} />
            )}

            <section className="border border-border bg-surface p-5">
              <div className="flex items-center gap-2 text-[12px] font-mono font-bold uppercase tracking-widest text-text2 mb-3">
                <FileText size={16} />
                Recommendations
              </div>
              <ul className="list-disc pl-5 space-y-2 text-[13px] text-text2 leading-relaxed">
                {recommendations.map(line => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </section>

            <section className="border border-border bg-surface p-5">
              <div className="text-[12px] font-mono font-bold uppercase tracking-widest text-text2 mb-3">
                Reports shared with you
              </div>
              {(reportsQuery.data?.length ?? 0) === 0 ? (
                <EmptyState
                  icon={<FileText size={22} />}
                  title="No PDFs yet"
                  description="Your care team will publish summaries here after review."
                />
              ) : (
                <ul className="space-y-3">
                  {reportsQuery.data!.map(r => (
                    <li key={r.id} className="border border-border2 bg-bg px-4 py-3">
                      <div className="font-mono text-[11px] text-teal">{r.title}</div>
                      <p className="text-[13px] text-text2 mt-2">{r.summarySnippet}</p>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </>
        )}
      </main>
    </div>
  )
}
