import type { Patient, Scan } from '@/types'
import PatientRow from './PatientRow'
import SkeletonRow from '@/components/shared/SkeletonRow'

interface PatientTableProps {
  patients: Patient[]
  scansMap: Record<string, Scan[]>
  loading?: boolean
}

const COLUMNS = ['Patient', 'Diagnosis', 'Scans', 'Last MRI', 'Volume (latest)', 'Status', '']

export default function PatientTable({ patients, scansMap, loading = false }: PatientTableProps) {
  return (
    <table className="w-full border-collapse">
      <thead>
        <tr className="border-b border-border">
          {COLUMNS.map(col => (
            <th
              key={col}
              className="text-left px-3 py-2 text-[10px] font-mono font-bold tracking-widest uppercase text-text3"
            >
              {col}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {loading
          ? Array.from({ length: 5 }).map((_, i) => <SkeletonRow key={i} />)
          : patients.map(patient => {
              const scans = (scansMap[patient.id] ?? []).sort(
                (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
              )
              const latest = scans[scans.length - 1]
              const previous = scans[scans.length - 2]
              return (
                <PatientRow
                  key={patient.id}
                  patient={patient}
                  latestScan={latest}
                  previousScan={previous}
                />
              )
            })}
      </tbody>
    </table>
  )
}
