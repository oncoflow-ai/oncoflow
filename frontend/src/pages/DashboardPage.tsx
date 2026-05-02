import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useAuthStore } from '@/store/authStore'
import TopNav from '@/components/layout/TopNav'
import BackendOperatorWorkspace from '@/components/dashboard/BackendOperatorWorkspace'
import LongitudinalComparisonPanel from '@/components/dashboard/LongitudinalComparisonPanel'
import PatientTable from '@/components/patient/PatientTable'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import { getPatients } from '@/api/patients'
import { getScans } from '@/api/scans'
import type { Scan } from '@/types'
import { Users } from 'lucide-react'

export default function DashboardPage() {
  const [search, setSearch] = useState('')
  const physician = useAuthStore(s => s.physician)

  const { data: patients = [], isLoading, isError, refetch } = useQuery({
    queryKey: ['patients'],
    queryFn: getPatients,
  })

  // Fetch scans for all patients in parallel
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

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    if (!q) return patients
    return patients.filter(p =>
      p.name.toLowerCase().includes(q) || p.id.toLowerCase().includes(q)
    )
  }, [patients, search])

  const scansMap = scanQueries.data?.map ?? {}
  const scanFetchFailed = (scanQueries.data?.failedPatientIds.length ?? 0) > 0

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <TopNav
        showSearch
        searchValue={search}
        onSearchChange={setSearch}
        cta={
          <button className="border border-teal text-teal font-mono font-bold text-[12px] tracking-widest uppercase px-3.5 py-1.5 hover:bg-teal/5 transition-colors">
            + New Patient
          </button>
        }
      />

      <main className="flex-1 px-5 py-6">
        <div className="flex flex-col gap-6">
          <BackendOperatorWorkspace />

          <LongitudinalComparisonPanel />

          <section className="bg-surface border border-border">
            {isError && (
              <ErrorBanner
                message="Failed to load patients."
                onRetry={() => refetch()}
              />
            )}
            {scanFetchFailed && (
              <ErrorBanner
                message="Some scan histories could not be loaded."
                onRetry={() => scanQueries.refetch()}
              />
            )}

            <div className="px-5 py-3.5 border-b border-border flex items-center justify-between">
              <span className="text-[11px] font-mono text-text3 uppercase tracking-widest">
                Mock roster · {filtered.length} patient{filtered.length !== 1 ? 's' : ''} · {physician?.name ?? 'Dr.'} · Oncology
              </span>
              <span className="text-[11px] font-mono text-text3">Secondary dataset for UI scaffolding</span>
            </div>

            {!isLoading && filtered.length === 0 ? (
              <EmptyState
                icon={<Users size={28} />}
                title="No patients found"
                description={search ? 'Try a different name or ID.' : 'No patients assigned to your account yet.'}
              />
            ) : (
              <PatientTable
                patients={filtered}
                scansMap={scansMap}
                loading={isLoading || scanQueries.isLoading}
              />
            )}
          </section>
        </div>
      </main>
    </div>
  )
}
