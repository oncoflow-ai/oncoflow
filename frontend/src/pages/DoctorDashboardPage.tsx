import { useState, useMemo } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import TopNav from '@/components/layout/TopNav'
import PatientTable from '@/components/patient/PatientTable'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import { getPatients, createPatient } from '@/api/patients'
import { getScans } from '@/api/scans'
import type { Scan } from '@/types'
import { Users } from 'lucide-react'
import { useAuthStore } from '@/store/authStore'

export default function DoctorDashboardPage() {
  const [search, setSearch] = useState('')
  const [modalOpen, setModalOpen] = useState(false)
  const [formName, setFormName] = useState('')
  const [formDob, setFormDob] = useState('')
  const [formDx, setFormDx] = useState('')
  const [formLoc, setFormLoc] = useState('')
  const user = useAuthStore(s => s.user)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: patients = [], isLoading, isError, refetch } = useQuery({
    queryKey: ['patients'],
    queryFn: getPatients,
  })

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

  const createMutation = useMutation({
    mutationFn: () =>
      createPatient({
        name: formName,
        dob: formDob,
        diagnosis: formDx,
        diagnosisLocation: formLoc,
        assignedPhysicianId: user?.id ?? 'DR-001',
      }),
    onSuccess: patient => {
      queryClient.invalidateQueries({ queryKey: ['patients'] })
      setModalOpen(false)
      setFormName('')
      setFormDob('')
      setFormDx('')
      setFormLoc('')
      navigate(`/doctor/patients/${patient.id}`)
    },
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
          <button
            type="button"
            onClick={() => setModalOpen(true)}
            className="border border-teal text-teal font-mono font-bold text-[12px] tracking-widest uppercase px-3.5 py-1.5 hover:bg-teal/5 transition-colors"
          >
            + New Patient
          </button>
        }
      />

      <main className="flex-1 px-5 py-6">
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
              Patient roster · {filtered.length} patient{filtered.length !== 1 ? 's' : ''} · {user?.name ?? 'Doctor'} · Oncology
            </span>
            <span className="text-[11px] font-mono text-text3 hidden sm:inline">Mock data · demo workflow</span>
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
      </main>

      {modalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
          <div className="bg-surface border border-border max-w-md w-full p-6 shadow-xl">
            <h2 className="font-sans font-bold text-[18px] text-text1 mb-4">New patient (mock)</h2>
            <div className="space-y-3">
              <label className="block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                Full name
                <input
                  value={formName}
                  onChange={e => setFormName(e.target.value)}
                  className="mt-1 w-full bg-bg border border-border2 px-3 py-2 text-[14px] text-text1"
                />
              </label>
              <label className="block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                DOB (YYYY-MM-DD)
                <input
                  value={formDob}
                  onChange={e => setFormDob(e.target.value)}
                  placeholder="1994-07-22"
                  className="mt-1 w-full bg-bg border border-border2 px-3 py-2 text-[14px] text-text1"
                />
              </label>
              <label className="block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                Diagnosis
                <input
                  value={formDx}
                  onChange={e => setFormDx(e.target.value)}
                  className="mt-1 w-full bg-bg border border-border2 px-3 py-2 text-[14px] text-text1"
                />
              </label>
              <label className="block text-[11px] font-mono font-bold uppercase tracking-widest text-text3">
                Location
                <input
                  value={formLoc}
                  onChange={e => setFormLoc(e.target.value)}
                  className="mt-1 w-full bg-bg border border-border2 px-3 py-2 text-[14px] text-text1"
                />
              </label>
            </div>
            {createMutation.isError && (
              <p className="text-danger text-[12px] font-mono mt-3">Unable to create patient.</p>
            )}
            <div className="flex justify-end gap-2 mt-6">
              <button
                type="button"
                onClick={() => setModalOpen(false)}
                className="px-4 py-2 text-[12px] font-mono uppercase tracking-widest text-text2 border border-border2"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={
                  createMutation.isPending || !formName.trim() || !formDob.trim() || !formDx.trim() || !formLoc.trim()
                }
                onClick={() => createMutation.mutate()}
                className="px-4 py-2 bg-teal text-black text-[12px] font-mono font-bold uppercase tracking-widest disabled:opacity-40"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
