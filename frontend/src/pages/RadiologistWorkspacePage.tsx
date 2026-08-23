import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import TopNav from '@/components/layout/TopNav'
import PatientTable from '@/components/patient/PatientTable'
import BackendOperatorWorkspace from '@/components/dashboard/BackendOperatorWorkspace'
import LongitudinalComparisonPanel from '@/components/dashboard/LongitudinalComparisonPanel'
import ErrorBanner from '@/components/shared/ErrorBanner'
import EmptyState from '@/components/shared/EmptyState'
import { getPatients } from '@/api/patients'
import { getScans } from '@/api/scans'
import { listStudies, submitComparison } from '@/api/backendWorkspace'
import { saveMriAnalysisReport } from '@/api/reports'
import type { Patient, Scan } from '@/types'
import { Users } from 'lucide-react'

export default function RadiologistWorkspacePage() {
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null)
  const [autoMsg, setAutoMsg] = useState<string | null>(null)
  const queryClient = useQueryClient()
  const navigate = useNavigate()

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

  const autoCompareMutation = useMutation({
    mutationFn: async () => {
      const studies = await listStudies()
      let eligible = studies.filter(s => s.hasResults)
      if (selectedPatient?.linkedStudyIds?.length) {
        eligible = eligible.filter(s => selectedPatient.linkedStudyIds!.includes(s.studyId))
      }
      const sorted = [...eligible].sort((a, b) => {
        const ta = new Date(a.acquiredAt ?? a.createdAt).getTime()
        const tb = new Date(b.acquiredAt ?? b.createdAt).getTime()
        return ta - tb
      })
      if (sorted.length < 2) return null
      const baseline = sorted[0]
      const followup = sorted[sorted.length - 1]
      if (baseline.studyId === followup.studyId) return null
      return submitComparison({
        baselineStudyId: baseline.studyId,
        followupStudyId: followup.studyId,
      })
    },
    onSuccess: res => {
      if (res) {
        setAutoMsg(`Automatic longitudinal comparison completed (${res.comparisonId.slice(0, 8)}…).`)
      }
      queryClient.invalidateQueries({ queryKey: ['backend-studies'] })
    },
    onError: () => {
      setAutoMsg('Segmentation finished, but automatic comparison failed — ensure at least two studies have results.')
    },
  })

  function handleJobTerminal(payload: {
    studyId: string
    status: 'completed' | 'failed'
    mode: 'nifti' | 'dicom-zip' | 'class-demo'
  }) {
    if (payload.status !== 'completed') {
      setAutoMsg(null)
      return
    }
    queryClient.invalidateQueries({ queryKey: ['backend-studies'] })
    if (selectedPatient) {
      void autoCompareMutation.mutateAsync().catch(() => {})
      saveMriAnalysisReport(selectedPatient.id, payload.studyId)
      queryClient.invalidateQueries({ queryKey: ['reports', selectedPatient.id] })
      navigate(`/patients/${selectedPatient.id}/results/${payload.studyId}`)
      return
    }
    void autoCompareMutation.mutateAsync().catch(() => {})
  }

  const scansMap = scanQueries.data?.map ?? {}
  const scanFetchFailed = (scanQueries.data?.failedPatientIds.length ?? 0) > 0

  const prefilledLabel = useMemo(() => {
    if (!selectedPatient) return ''
    return `${selectedPatient.id} · ${selectedPatient.name}`
  }, [selectedPatient])

  return (
    <div className="min-h-screen bg-bg flex flex-col">
      <TopNav />

      <main className="flex-1 px-5 py-6 flex flex-col gap-6">
        <section className="bg-surface border border-border">
          <div className="px-5 py-3.5 border-b border-border flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <span className="text-[11px] font-mono text-text3 uppercase tracking-widest">
              Select patient · upload routes segmentation · longitudinal runs when ≥2 studies exist
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
              selectedPatientId={selectedPatient?.id ?? null}
              onSelectPatient={setSelectedPatient}
            />
          )}
        </section>

        {!selectedPatient ? (
          <EmptyState
            icon={<Users size={28} />}
            title="Choose a patient row"
            description="Selecting a patient prefills the upload label and scopes automatic longitudinal comparison when linkage IDs exist."
          />
        ) : (
          <>
            {autoMsg && (
              <div className="rounded border border-teal/25 bg-teal/5 px-4 py-3 font-mono text-[12px] text-text2">
                {autoMsg}
              </div>
            )}
            <BackendOperatorWorkspace
              headingEyebrow="Radiologist workspace"
              headingTitle="Upload MRI — segmentation pipeline"
              headingDescription="After ingestion completes, we attempt an automatic longitudinal comparison between the earliest and latest backend studies that have stored results (optionally filtered by this patient's linkedStudyIds when configured)."
              prefilledSourceLabel={prefilledLabel}
              onJobReachedTerminal={handleJobTerminal}
            />
            <LongitudinalComparisonPanel
              restrictToStudyIds={selectedPatient.linkedStudyIds}
              scopeNote={
                selectedPatient.linkedStudyIds?.length
                  ? undefined
                  : 'Showing all demo-backend studies. Configure linkedStudyIds on the mock patient to narrow comparisons.'
              }
            />
          </>
        )}
      </main>
    </div>
  )
}
