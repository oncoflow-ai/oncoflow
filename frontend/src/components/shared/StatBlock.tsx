import DeltaTag from './DeltaTag'
import { cn } from '@/lib/utils'

interface StatBlockProps {
  label: string
  value: string
  delta?: number | null
  deltaUnit?: string
  badge?: React.ReactNode
  className?: string
}

export default function StatBlock({ label, value, delta, deltaUnit, badge, className }: StatBlockProps) {
  return (
    <div className={cn('bg-surface px-4 py-[18px]', className)}>
      <div className="text-[10px] font-mono font-bold tracking-widest uppercase text-text3 mb-2.5">
        {label}
      </div>
      <div className="font-mono text-[26px] font-bold text-text1 leading-none">
        {value}
      </div>
      <div className="mt-2">
        {badge ?? (delta !== undefined ? (
          <DeltaTag value={delta ?? null} unit={deltaUnit} />
        ) : null)}
      </div>
    </div>
  )
}
