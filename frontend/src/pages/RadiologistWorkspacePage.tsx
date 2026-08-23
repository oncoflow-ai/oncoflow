import { useQuery } from '@tanstack/react-query'
import TopNav from '@/components/layout/TopNav'
import PatientTable from '@/components/patient/PatientTable'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import { getPatients } from '@/api/patients'
import { getScans } from '@/api/scans'
import type { Scan } from '@/types'
import { Users } from 'lucide-react'

export default function RadiologistWorkspacePage() {
  const patientsQuery = useQuery({
    queryKey: ['patients'],
    queryFn: getPatients,
  })

  const patients = patientsQuery.data ?? []

  const scanQueries = useQuery({
    queryKey: ['all-scans', patients.map(p => p.id)],
    queryFn: async () => {
      const results = await Promise.allSettled(patients.map(p => getScans(p.id)))
      const map: Record<string, Scan[]> = {}
      const failedPatientIds: string[] = []

      patients.forEach((patient, index) => {
        const result = results[index]
        if (result.status === 'fulfilled') {
          map[patient.id] = result.value
          return
        }
        failedPatientIds.push(patient.id)
      })

      return { map, failedPatientIds }
    },
    enabled: patients.length > 0,
  })

  const scansMap = scanQueries.data?.map ?? {}
  const scanFetchFailed = (scanQueries.data?.failedPatientIds.length ?? 0) > 0

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <TopNav />

      <main className="flex-1 px-5 py-6 flex flex-col gap-6">
        <section className="bg-surface border border-border">
          <div className="px-5 py-3.5 border-b border-border flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <span className="text-[11px] font-mono text-text3 uppercase tracking-widest">
              Select a patient to open their chart and MRI workspace
            </span>
          </div>
          {patientsQuery.isError && (
            <div className="p-5">
              <ErrorBanner message="Failed to load patients." onRetry={() => patientsQuery.refetch()} />
            </div>
          )}
          {scanFetchFailed && (
            <div className="px-5 pt-5">
              <ErrorBanner message="Some scan histories could not be loaded." onRetry={() => scanQueries.refetch()} />
            </div>
          )}
          {!patientsQuery.isLoading && patients.length === 0 ? (
            <EmptyState icon={<Users size={28} />} title="No patients" description="Add patients from the doctor roster first." />
          ) : (
            <PatientTable
              patients={patients}
              scansMap={scansMap}
              loading={patientsQuery.isLoading || scanQueries.isLoading}
              rowMode="select"
            />
          )}
        </section>
      </main>
    </div>
  )
}
