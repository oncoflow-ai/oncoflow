import { useState } from 'react'
import type { Scan } from '@/types'
import {
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Eye,
  EyeOff,
  Layers,
  Maximize2,
  Pencil,
  RotateCcw,
  Ruler,
  Trash2,
  ZoomIn,
} from 'lucide-react'
import { cn, formatVolume } from '@/lib/utils'

interface MriWorkspaceProps {
  scan: Scan
}

type Tool = 'brush' | 'ruler' | 'crosshair' | 'delete'
type OverlayMode = 'all' | 'contour' | 'none'

const FRAMES = [72, 78, 82, 86, 92]

export default function MriWorkspace({ scan }: MriWorkspaceProps) {
  const [activeTool, setActiveTool] = useState<Tool>('brush')
  const [slice, setSlice] = useState(82)
  const [zoomLevel, setZoomLevel] = useState<number>(1)
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('all')
  const [showRuler, setShowRuler] = useState(true)
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)

  const frameSlice = FRAMES.reduce((best, current) =>
    Math.abs(current - slice) < Math.abs(best - slice) ? current : best
  )
  const frameSrc = `/demo-assets/p01-t1c-seg-slice-${frameSlice}.png`

  function stepSlice(delta: number) {
    setSlice(prev => Math.min(92, Math.max(72, prev + delta)))
  }

  function toggleZoom() {
    setZoomLevel(prev => (prev === 1 ? 1.4 : prev === 1.4 ? 2 : 1))
  }

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = Math.round(((e.clientX - rect.left) / rect.width) * 256)
    const y = Math.round(((e.clientY - rect.top) / rect.height) * 256)
    setMousePos({ x, y })
  }

  return (
    <aside className="w-[340px] xl:w-[400px] shrink-0 bg-[#060810] border-l border-border flex flex-col select-none">
      {/* Header */}
      <div className="px-4 py-3 border-b border-border bg-surface/30">
        <div className="flex items-center justify-between">
          <div className="text-[10px] font-mono font-bold tracking-widest uppercase text-teal flex items-center gap-1.5">
            <Layers size={12} />
            MRI Interactive Viewer
          </div>
          <div className="border border-teal/30 bg-teal/10 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-teal">
            AI Segmentation
          </div>
        </div>
        <div className="text-[14px] font-sans font-semibold text-text1 mt-1">{scan.studyLabel}</div>
        <div className="font-mono text-[10px] text-text3 mt-0.5 flex items-center gap-2">
          <span>{scan.sequence}</span>
          <span>•</span>
          <span>{scan.plane}</span>
          <span>•</span>
          <span>{scan.sliceCount} SL</span>
          <span>•</span>
          <span>{scan.resolution}</span>
        </div>
      </div>

      {/* Main Interactive Viewer Viewport */}
      <div
        className="relative flex-1 bg-black flex flex-col justify-between overflow-hidden cursor-crosshair min-h-[340px]"
        onWheel={e => {
          e.preventDefault()
          stepSlice(e.deltaY > 0 ? 1 : -1)
        }}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setMousePos(null)}
      >
        {/* HUD Top Bar */}
        <div className="absolute top-2.5 inset-x-3 z-20 flex items-center justify-between pointer-events-none">
          <div className="font-mono text-[10px] bg-bg/85 border border-border2 px-2 py-0.5 text-text2">
            Axial · Slice {String(frameSlice).padStart(3, '0')} / {scan.sliceCount}
          </div>
          <div className="font-mono text-[10px] bg-bg/85 border border-border2 px-2 py-0.5 text-teal">
            Zoom: {zoomLevel}x
          </div>
        </div>

        {/* Center Image Container */}
        <div className="relative flex-1 flex items-center justify-center p-3 overflow-hidden">
          {/* Viewport Crosshair Guides */}
          {activeTool === 'crosshair' && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
              <div className="absolute left-0 right-0 h-px bg-teal/20" />
              <div className="absolute top-0 bottom-0 w-px bg-teal/20" />
            </div>
          )}

          <div
            className="relative aspect-square w-full max-w-[280px] xl:max-w-[320px] overflow-hidden rounded border border-border/70 bg-black shadow-2xl transition-transform duration-150 ease-out"
            style={{ transform: `scale(${zoomLevel})` }}
          >
            {/* Real MRI Slice Image with AI Segmentation Overlay */}
            <img
              src={frameSrc}
              alt={`Axial MRI scan slice ${frameSlice}`}
              className={cn(
                'h-full w-full object-contain pointer-events-none',
                overlayMode === 'none' && 'grayscale contrast-125 brightness-110'
              )}
            />

            {/* Caliper measurement annotation line when ruler tool is active or enabled */}
            {showRuler && (
              <div className="absolute pointer-events-none inset-0 flex items-center justify-center">
                <div
                  className="relative"
                  style={{
                    width: 72,
                    top: -12,
                    left: 24,
                    borderTop: '1.5px dashed #0DC5A0',
                  }}
                >
                  <div className="absolute -left-1 -top-1 w-2 h-2 rounded-full bg-teal shadow-[0_0_6px_#0DC5A0]" />
                  <div className="absolute -right-1 -top-1 w-2 h-2 rounded-full bg-teal shadow-[0_0_6px_#0DC5A0]" />
                  <div className="absolute -top-5 left-1/2 -translate-x-1/2 font-mono text-[10px] font-bold text-teal bg-bg/90 px-1.5 py-0.2 rounded border border-teal/40 shadow">
                    {scan.maxDiameterMm} mm
                  </div>
                </div>
              </div>
            )}

            {/* Subtly Vignette Overlay */}
            <div className="pointer-events-none absolute inset-0 shadow-[inset_0_0_35px_rgba(0,0,0,0.7)]" />
          </div>
        </div>

        {/* HUD Bottom Info Bar */}
        <div className="px-3 py-2 z-20 flex items-center justify-between border-t border-border/60 bg-bg/90 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span className="font-mono text-[10px] text-rose-200">Tumor</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span className="font-mono text-[10px] text-cyan-200">Edema</span>
            </div>
          </div>
          <div className="font-mono text-[10px] text-text2">
            Vol: <span className="text-text1 font-bold">{formatVolume(scan.volumeMm3)} mm³</span>
          </div>
          {mousePos && (
            <div className="font-mono text-[9px] text-text3">
              X:{mousePos.x} Y:{mousePos.y}
            </div>
          )}
        </div>
      </div>

      {/* Slice Scrubber & Quick Controls */}
      <div className="p-3 border-t border-border bg-surface/50 space-y-2.5">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => stepSlice(-1)}
            aria-label="Previous slice"
            className="w-7 h-7 bg-surface2 border border-border2 hover:border-teal hover:text-teal flex items-center justify-center text-text2 transition-colors"
          >
            <ChevronLeft size={14} />
          </button>
          <input
            aria-label="MRI slice"
            type="range"
            min={72}
            max={92}
            value={slice}
            onChange={e => setSlice(Number(e.target.value))}
            className="h-1.5 flex-1 accent-teal cursor-pointer"
          />
          <button
            type="button"
            onClick={() => stepSlice(1)}
            aria-label="Next slice"
            className="w-7 h-7 bg-surface2 border border-border2 hover:border-teal hover:text-teal flex items-center justify-center text-text2 transition-colors"
          >
            <ChevronRight size={14} />
          </button>
          <span className="font-mono text-[11px] text-text2 min-w-[62px] text-right">
            {frameSlice} / {scan.sliceCount}
          </span>
        </div>
      </div>

      {/* Interactive Tool Strip */}
      <div className="px-3 py-2.5 border-t border-border bg-[#090d16] flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setActiveTool('brush')}
            title="Brush annotation"
            aria-label="Brush annotation"
            className={cn(
              'w-8 h-8 flex items-center justify-center border transition-colors',
              activeTool === 'brush'
                ? 'bg-teal/15 border-teal text-teal'
                : 'bg-surface2 border-border2 text-text2 hover:border-text3'
            )}
          >
            <Pencil size={13} />
          </button>

          <button
            type="button"
            onClick={() => {
              setActiveTool('ruler')
              setShowRuler(true)
            }}
            title="Measure diameter"
            aria-label="Measure diameter"
            className={cn(
              'w-8 h-8 flex items-center justify-center border transition-colors',
              activeTool === 'ruler'
                ? 'bg-teal/15 border-teal text-teal'
                : 'bg-surface2 border-border2 text-text2 hover:border-text3'
            )}
          >
            <Ruler size={13} />
          </button>

          <button
            type="button"
            onClick={() => setActiveTool('crosshair')}
            title="Crosshair inspect"
            aria-label="Crosshair inspect"
            className={cn(
              'w-8 h-8 flex items-center justify-center border transition-colors',
              activeTool === 'crosshair'
                ? 'bg-teal/15 border-teal text-teal'
                : 'bg-surface2 border-border2 text-text2 hover:border-text3'
            )}
          >
            <Crosshair size={13} />
          </button>

          <button
            type="button"
            onClick={() => setOverlayMode(prev => (prev === 'all' ? 'none' : 'all'))}
            title={overlayMode === 'all' ? 'Hide AI Mask' : 'Show AI Mask'}
            aria-label={overlayMode === 'all' ? 'Hide AI Mask' : 'Show AI Mask'}
            className={cn(
              'w-8 h-8 flex items-center justify-center border transition-colors',
              overlayMode === 'all'
                ? 'bg-teal/15 border-teal text-teal'
                : 'bg-surface2 border-border2 text-text3 hover:border-text3'
            )}
          >
            {overlayMode === 'all' ? <Eye size={13} /> : <EyeOff size={13} />}
          </button>

          <button
            type="button"
            onClick={toggleZoom}
            title={`Zoom (${zoomLevel}x)`}
            aria-label="Zoom"
            className="w-8 h-8 flex items-center justify-center bg-surface2 border border-border2 text-text2 hover:border-teal hover:text-teal transition-colors"
          >
            <ZoomIn size={13} />
          </button>
        </div>

        <button
          type="button"
          onClick={() => {
            setSlice(82)
            setZoomLevel(1)
            setOverlayMode('all')
            setShowRuler(true)
            setActiveTool('brush')
          }}
          title="Reset viewer"
          aria-label="Reset viewer"
          className="w-8 h-8 flex items-center justify-center bg-surface2 border border-border2 text-text3 hover:text-danger hover:border-danger transition-colors"
        >
          <RotateCcw size={13} />
        </button>
      </div>
    </aside>
  )
}
