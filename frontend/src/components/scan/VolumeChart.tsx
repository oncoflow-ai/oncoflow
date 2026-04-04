import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  TooltipProps,
} from 'recharts'
import type { Scan } from '@/types'
import { formatDate, formatVolume } from '@/lib/utils'

interface VolumeChartProps {
  scans: Scan[]
}

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-surface2 border border-border2 px-3 py-2 font-mono text-[11px]">
      <div className="text-text3 mb-1">{label}</div>
      <div className="text-teal font-bold">{formatVolume(payload[0].value as number)} mm³</div>
    </div>
  )
}

export default function VolumeChart({ scans }: VolumeChartProps) {
  const data = scans.map(s => ({
    date: formatDate(s.date),
    volume: s.volumeMm3,
  }))

  return (
    <div className="bg-surface border border-border p-5">
      <div className="flex items-baseline justify-between mb-1">
        <span className="text-[12px] font-mono font-bold tracking-widest uppercase text-text2">
          Tumor Volume Trajectory
        </span>
        <span className="text-[11px] font-mono text-text3">
          mm³ · {scans.length} data point{scans.length !== 1 ? 's' : ''}
        </span>
      </div>
      <div className="mt-4 h-[120px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 4, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="tealGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0DC5A0" stopOpacity={0.15} />
                <stop offset="100%" stopColor="#0DC5A0" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#252A3A" strokeDasharray="0" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: '#4E566A', fontSize: 10, fontFamily: 'Geist Mono' }}
              axisLine={{ stroke: '#252A3A' }}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: '#4E566A', fontSize: 10, fontFamily: 'Geist Mono' }}
              axisLine={false}
              tickLine={false}
              tickFormatter={v => `${Math.round(v / 1000)}k`}
              width={32}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="volume"
              stroke="#0DC5A0"
              strokeWidth={2}
              fill="url(#tealGrad)"
              dot={{ fill: '#0B0D12', stroke: '#0DC5A0', strokeWidth: 2, r: 4 }}
              activeDot={{ fill: '#0DC5A0', r: 5 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
