import { cn } from '@/lib/utils'

interface DeltaTagProps {
  value: number | null
  unit?: string
  className?: string
}

export default function DeltaTag({ value, unit = '%', className }: DeltaTagProps) {
  if (value === null) {
    return (
      <span className={cn('inline-flex items-center font-mono text-[10px] bg-surface2 text-text2 px-1.5 py-0.5', className)}>
        — Baseline
      </span>
    )
  }
  const isPositive = value > 0
  return (
    <span className={cn(
      'inline-flex items-center font-mono text-[10px] px-1.5 py-0.5',
      isPositive
        ? 'bg-danger/10 text-danger'
        : 'bg-positive/10 text-positive',
      className
    )}>
      {isPositive ? '▲' : '▼'} {Math.abs(value)}{unit}
    </span>
  )
}
