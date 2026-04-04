import { useState } from 'react'
import type { Scan } from '@/types'
import { Pencil, Ruler, Trash2 } from 'lucide-react'
import { cn } from '@/lib/utils'

interface MriWorkspaceProps {
  scan: Scan
}

type Tool = 'brush' | 'ruler' | 'delete'

const TOOLS: { id: Tool; icon: React.ReactNode; label: string }[] = [
  { id: 'brush', icon: <Pencil size={13} />, label: 'Brush annotation' },
  { id: 'ruler', icon: <Ruler size={13} />, label: 'Measure' },
  { id: 'delete', icon: <Trash2 size={13} />, label: 'Delete annotation' },
]

export default function MriWorkspace({ scan }: MriWorkspaceProps) {
  const [activeTool, setActiveTool] = useState<Tool>('brush')

  return (
    <aside className="w-[280px] shrink-0 bg-[#060810] border-l border-border flex flex-col">
      {/* Header */}
      <div className="px-4 py-3.5 border-b border-border">
        <div className="text-[10px] font-mono font-bold tracking-widest uppercase text-text3 mb-1">
          MRI Workspace
        </div>
        <div className="text-[13px] font-sans font-medium text-text1">{scan.studyLabel}</div>
        <div className="font-mono text-[10px] text-text3 mt-0.5">
          {scan.sequence} · {scan.plane} · {scan.sliceCount} SL · {scan.resolution}
        </div>
      </div>

      {/* Viewer */}
      <div className="flex-1 flex items-center justify-center relative">
        {/* Crosshair lines */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="absolute left-0 right-0 h-px bg-teal/10" />
          <div className="absolute top-0 bottom-0 w-px bg-teal/10" />
        </div>

        <div className="flex flex-col items-center gap-3">
          {/* MRI slice mockup */}
          <div className="relative">
            <div
              className="w-[180px] h-[180px] rounded-full border border-[#1a2030]"
              style={{
                background: `
                  radial-gradient(ellipse at 30% 40%, #2a3548 0%, transparent 50%),
                  radial-gradient(ellipse at 65% 55%, #1e2a3a 0%, transparent 45%),
                  radial-gradient(ellipse at 50% 50%, #141e2f 0%, #080c14 100%)
                `,
                boxShadow: '0 0 40px rgba(0,0,0,0.8), inset 0 0 30px rgba(0,0,0,0.5)',
              }}
            >
              {/* Tumor overlay */}
              <div
                className="absolute"
                style={{
                  width: 34, height: 28,
                  top: 68, left: 73,
                  borderRadius: '50%',
                  background: 'rgba(224, 82, 82, 0.22)',
                  border: '1.5px solid rgba(224, 82, 82, 0.5)',
                  boxShadow: '0 0 12px rgba(224, 82, 82, 0.2)',
                }}
              />
              {/* Ruler annotation */}
              <div
                className="absolute"
                style={{ top: 66, left: 69, width: 40, borderTop: '1px dashed rgba(13,197,160,0.7)' }}
              />
              <div
                className="absolute font-mono text-[8px] text-teal"
                style={{ top: 56, left: 70 }}
              >
                {scan.maxDiameterMm} mm
              </div>
            </div>
          </div>

          {/* Slice navigation — cosmetic only */}
          <div className="flex items-center gap-2.5">
            <button className="w-6 h-6 bg-surface2 border border-border2 flex items-center justify-center text-text2 text-[11px]">
              ‹
            </button>
            <span className="font-mono text-[10px] text-text3">
              {Math.floor(scan.sliceCount / 2).toString().padStart(3, '0')} / {scan.sliceCount}
            </span>
            <button className="w-6 h-6 bg-surface2 border border-border2 flex items-center justify-center text-text2 text-[11px]">
              ›
            </button>
          </div>
        </div>
      </div>

      {/* Toolbar */}
      <div className="px-4 py-3 border-t border-border flex items-center justify-between">
        <div className="flex gap-1.5">
          {TOOLS.map(tool => (
            <button
              key={tool.id}
              onClick={() => setActiveTool(tool.id)}
              title={tool.label}
              className={cn(
                'w-8 h-8 flex items-center justify-center border transition-colors',
                activeTool === tool.id
                  ? 'bg-teal/10 border-teal text-teal'
                  : 'bg-surface2 border-border2 text-text2 hover:border-text3'
              )}
            >
              {tool.icon}
            </button>
          ))}
        </div>
        <button className="border border-teal text-teal font-mono text-[11px] font-bold tracking-widest uppercase px-3 py-1.5 hover:bg-teal/5 transition-colors">
          ↓ PDF
        </button>
      </div>
    </aside>
  )
}
