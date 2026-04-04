import type { Scan } from '@/types'
import { formatDate, formatVolume } from '@/lib/utils'
import { ScanLine } from 'lucide-react'

interface ScanRowProps {
  scan: Scan
  index: number
}

export default function ScanRow({ scan, index }: ScanRowProps) {
  return (
    <div className="flex items-center gap-3.5 py-3 border-b border-border last:border-b-0 last:pb-0 first:pt-0">
      <span className="font-mono text-[10px] text-text3 w-5 shrink-0 text-right">
        {String(index).padStart(2, '0')}
      </span>
      <div className="w-8 h-8 bg-surface3 border border-border2 flex items-center justify-center shrink-0 text-teal">
        <ScanLine size={14} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-sans font-medium text-[13px] text-text1">{formatDate(scan.date)}</div>
        <div className="font-mono text-[10px] text-text3 mt-0.5 truncate">
          {scan.studyLabel} · {scan.sequence} · {scan.plane} · {scan.sliceCount} slices · {scan.resolution}
        </div>
      </div>
      <div className="text-right shrink-0">
        <div className="font-mono text-[13px] text-text1">{formatVolume(scan.volumeMm3)} mm³</div>
        <div className="font-mono text-[10px] text-text3 mt-0.5">Ø {scan.maxDiameterMm} mm</div>
      </div>
      <span className={`inline-flex items-center gap-1 font-mono text-[10px] font-bold px-2 py-0.5 shrink-0 ${
        scan.isAnnotated
          ? 'bg-teal/10 border border-teal/25 text-teal'
          : 'bg-surface3 border border-border2 text-text3'
      }`}>
        {scan.isAnnotated ? '✓ ANNOTATED' : '○ PENDING'}
      </span>
    </div>
  )
}
