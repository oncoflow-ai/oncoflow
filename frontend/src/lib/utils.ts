import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

/** shadcn/ui cn() helper — merge Tailwind classes */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Format a date string (ISO) to "MMM DD, YYYY" */
export function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString('en-US', {
    month: 'short',
    day: '2-digit',
    year: 'numeric',
  })
}

/** Format a volume in mm³ with thousands separator */
export function formatVolume(mm3: number): string {
  return mm3.toLocaleString('en-US')
}

/** Return sign-prefixed delta percentage string, e.g. "+12.3%" or "-4.5%" */
export function formatDelta(pct: number): string {
  const sign = pct > 0 ? '+' : ''
  return `${sign}${pct.toFixed(1)}%`
}

/** Calculate percent change between two volume measurements (positive = growth) */
export function calcVolumeDeltaPct(latestMm3: number, previousMm3: number): number {
  return Math.round(((latestMm3 - previousMm3) / previousMm3) * 1000) / 10
}
