import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ClipboardCheck } from 'lucide-react'
import TopNav from '@/components/layout/TopNav'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import { getPatients } from '@/api/patients'
import { getScans } from '@/api/scans'

export default function RadiologyReviewPage() {
  const { data: patients = [], isError, isLoading, refetch } = useQuery({
    queryKey: ['review-patients'],
    queryFn: getPatients,
  })

  const scanQueries = useQuery({
    queryKey: ['review-scans', patients.map(p => p.id)],
    queryFn: async () => {
      const results = await Promise.all(patients.map(p => getScans(p.id)))
      return results.flat()
    },
    enabled: patients.length > 0,
  })

  const reviewItems = useMemo(() => {
    const patientById = new Map(patients.map(patient => [patient.id, patient]))
    return (scanQueries.data ?? [])
      .filter(scan => !scan.isAnnotated)
      .map(scan => ({ scan, patient: patientById.get(scan.patientId) }))
      .filter(item => item.patient)
  }, [patients, scanQueries.data])

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <TopNav />

      <main className="flex-1 px-5 py-6">
        <section className="bg-surface border border-border">
          {isError && (
            <ErrorBanner
              message="Failed to load radiology review queue."
              onRetry={() => refetch()}
            />
          )}

          <div className="px-5 py-3.5 border-b border-border flex items-center justify-between">
            <span className="text-[11px] font-mono text-text3 uppercase tracking-widest">
              Radiology Review Queue
            </span>
            <span className="text-[11px] font-mono text-text3">
              {reviewItems.length} scan{reviewItems.length !== 1 ? 's' : ''} pending annotation
            </span>
          </div>

          {isLoading || scanQueries.isLoading ? (
            <div className="p-5 text-[13px] text-text3 font-mono">Loading review worklist...</div>
          ) : reviewItems.length === 0 ? (
            <EmptyState
              icon={<ClipboardCheck size={28} />}
              title="No scans pending review"
              description="All current scans have annotations."
            />
          ) : (
            <div className="divide-y divide-border">
              {reviewItems.map(({ scan, patient }) => (
                <div key={scan.id} className="px-5 py-4 grid gap-3 md:grid-cols-[1fr_auto] md:items-center">
                  <div>
                    <div className="text-text1 text-[14px] font-semibold">
                      {patient?.name} · {scan.studyLabel}
                    </div>
                    <div className="text-text3 text-[12px] font-mono mt-1">
                      {patient?.id} · {scan.modality} {scan.sequence} · {scan.sliceCount} slices · {scan.date}
                    </div>
                  </div>
                  <a
                    href={`/patients/${scan.patientId}`}
                    className="border border-teal text-teal font-mono font-bold text-[12px] tracking-widest uppercase px-3.5 py-2 hover:bg-teal/5 transition-colors text-center"
                  >
                    Open Study
                  </a>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
