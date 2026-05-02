import { useNavigate } from 'react-router-dom'
import type { Patient, Scan } from '@/types'
import DeltaTag from '@/components/shared/DeltaTag'
import { cn, formatDate, formatVolume, calcVolumeDeltaPct } from '@/lib/utils'

interface PatientRowProps {
  patient: Patient
  latestScan?: Scan
  previousScan?: Scan
  /** Opening patient chart vs selecting row (radiologist roster) */
  rowMode?: 'navigate' | 'select'
  navigateBase?: string
  selected?: boolean
  onActivate?: () => void
}

export default function PatientRow({
  patient,
  latestScan,
  previousScan,
  rowMode = 'navigate',
  navigateBase = '/doctor/patients',
  selected = false,
  onActivate,
}: PatientRowProps) {
  const navigate = useNavigate()

  function go() {
    if (rowMode === 'select') {
      onActivate?.()
      return
    }
    navigate(`${navigateBase.replace(/\/$/, '')}/${patient.id}`)
  }

  const volumeDelta = latestScan && previousScan
    ? calcVolumeDeltaPct(latestScan.volumeMm3, previousScan.volumeMm3)
    : null

  return (
    <tr
      className={cn(
        'border-b border-border cursor-pointer hover:bg-surface2 transition-colors',
        selected && 'bg-teal/5 ring-1 ring-inset ring-teal/25'
      )}
      onClick={go}
      onKeyDown={event => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          go()
        }
      }}
      tabIndex={0}
      role={rowMode === 'select' ? 'button' : 'link'}
      aria-current={selected ? 'true' : undefined}
      aria-label={rowMode === 'select' ? `Select patient ${patient.name}` : `Open patient ${patient.name}`}
    >
      <td className="px-3 py-3.5">
        <div className="font-sans font-semibold text-[15px] text-text1">{patient.name}</div>
        <div className="font-mono text-[10px] text-text3 mt-0.5">{patient.id}</div>
      </td>
      <td className="px-3 py-3.5">
        <div className="text-[12px] text-text2 font-sans leading-snug">
          {patient.diagnosis}<br />{patient.diagnosisLocation}
        </div>
      </td>
      <td className="px-3 py-3.5">
        <span className="inline-flex items-center gap-1 bg-surface3 border border-border2 text-teal font-mono text-[10px] font-bold px-2.5 py-1">
          ▣ {patient.scanCount} {patient.scanCount === 1 ? 'SCAN' : 'SCANS'}
        </span>
      </td>
      <td className="px-3 py-3.5">
        <div className="font-mono text-[11px] text-text2">{formatDate(patient.lastScanDate)}</div>
      </td>
      <td className="px-3 py-3.5">
        {latestScan ? (
          <>
            <div className="font-mono text-[13px] text-text1">
              {formatVolume(latestScan.volumeMm3)} <span className="text-[10px] text-text3">mm³</span>
            </div>
            <div className="mt-0.5">
              <DeltaTag value={volumeDelta} />
            </div>
          </>
        ) : (
          <span className="font-mono text-[10px] text-text3">No scans</span>
        )}
      </td>
      <td className="px-3 py-3.5">
        <div className="flex items-center gap-1.5">
          <span className={cn(
            'w-1.5 h-1.5 rounded-full',
            patient.status === 'active' ? 'bg-teal shadow-[0_0_6px_#0DC5A0]' : 'bg-amber'
          )} />
          <span className="text-[11px] text-text2 font-sans capitalize">{patient.status}</span>
        </div>
      </td>
      <td className="px-3 py-3.5 text-text3 text-[13px]">›</td>
    </tr>
  )
}
