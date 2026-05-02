import type { Scan } from '@/types'
import ScanRow from './ScanRow'

interface ImagingHistoryProps {
  scans: Scan[]
  selectedScanId?: string | null
  onSelectScan?: (scan: Scan) => void
}

export default function ImagingHistory({ scans, selectedScanId, onSelectScan }: ImagingHistoryProps) {
  const sorted = [...scans].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  return (
    <div className="bg-surface border border-border p-5">
      <div className="text-[12px] font-mono font-bold tracking-widest uppercase text-text2 mb-4">
        Imaging History
      </div>
      {sorted.map((scan, i) => (
        <ScanRow
          key={scan.id}
          scan={scan}
          index={sorted.length - i}
          selected={selectedScanId === scan.id}
          onSelect={onSelectScan ? () => onSelectScan(scan) : undefined}
        />
      ))}
    </div>
  )
}
