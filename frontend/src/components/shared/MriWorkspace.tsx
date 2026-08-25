import { useEffect, useRef, useState } from 'react'
import type { Scan } from '@/types'
import {
  ChevronLeft,
  ChevronRight,
  Crosshair,
  Eye,
  EyeOff,
  Layers,
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

interface Point {
  x: number // 0..256
  y: number // 0..256
}

interface Caliper {
  p1: Point
  p2: Point
}

type Tool = 'brush' | 'ruler' | 'crosshair'
type OverlayMode = 'all' | 'none'

const FRAMES = [72, 78, 82, 86, 92, 98, 102, 106, 110]
const MM_PER_PIXEL = 0.9375 // 240mm FOV / 256 matrix

function calcDistanceMm(p1: Point, p2: Point): number {
  const dx = p2.x - p1.x
  const dy = p2.y - p1.y
  return Math.sqrt(dx * dx + dy * dy) * MM_PER_PIXEL
}

function resolveDemoStageIndexFromScan(scan: Scan): number {
  if (typeof scan.demoStageIndex === 'number') {
    return scan.demoStageIndex
  }
  const vol = scan.volumeMm3
  if (vol <= 1800) return 4
  if (vol <= 2800) return 3
  if (vol <= 3500) return 1
  if (vol <= 6000) return 2
  return 0
}

export default function MriWorkspace({ scan }: MriWorkspaceProps) {
  const stageIndex = resolveDemoStageIndexFromScan(scan)
  const initialSlice = stageIndex > 0 ? 102 : 82
  const [activeTool, setActiveTool] = useState<Tool>('ruler')
  const [slice, setSlice] = useState(initialSlice)
  const [zoomLevel, setZoomLevel] = useState<number>(1)
  const [overlayMode, setOverlayMode] = useState<OverlayMode>('all')
  const [mousePos, setMousePos] = useState<{ x: number; y: number } | null>(null)

  // Caliper state (default positioned on the tumor lesion, scaled for shrinking lesion)
  const defaultCaliper = stageIndex > 0
    ? { p1: { x: 118, y: 120 }, p2: { x: 154, y: 120 } }
    : { p1: { x: 106, y: 118 }, p2: { x: 175, y: 118 } }

  const [caliper, setCaliper] = useState<Caliper | null>(defaultCaliper)
  const [isDrawingCaliper, setIsDrawingCaliper] = useState(false)
  const [draggedHandle, setDraggedHandle] = useState<'p1' | 'p2' | null>(null)

  // Update initial slice when scan changes
  useEffect(() => {
    setSlice(stageIndex > 0 ? 102 : 82)
    setCaliper(defaultCaliper)
  }, [scan.id, stageIndex])

  // Brush drawing strokes
  const [brushPaths, setBrushPaths] = useState<Point[][]>([])
  const [currentPath, setCurrentPath] = useState<Point[] | null>(null)

  const imageContainerRef = useRef<HTMLDivElement>(null)

  const frameSlice = FRAMES.reduce((best, current) =>
    Math.abs(current - slice) < Math.abs(best - slice) ? current : best
  )
  const frameSrc = `/demo-assets/demo-stage-${stageIndex}-slice-${frameSlice}.png`

  function stepSlice(delta: number) {
    setSlice(prev => Math.min(110, Math.max(72, prev + delta)))
  }

  function toggleZoom() {
    setZoomLevel(prev => (prev === 1 ? 1.4 : prev === 1.4 ? 2 : 1))
  }

  function getNormalizedCoords(e: React.MouseEvent<HTMLDivElement>): Point {
    if (!imageContainerRef.current) return { x: 128, y: 128 }
    const rect = imageContainerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(256, Math.round(((e.clientX - rect.left) / rect.width) * 256)))
    const y = Math.max(0, Math.min(256, Math.round(((e.clientY - rect.top) / rect.height) * 256)))
    return { x, y }
  }

  function handleMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    const pt = getNormalizedCoords(e)

    if (activeTool === 'ruler') {
      // Check if user clicked near existing caliper handles (within 14px)
      if (caliper) {
        const d1 = Math.hypot(pt.x - caliper.p1.x, pt.y - caliper.p1.y)
        const d2 = Math.hypot(pt.x - caliper.p2.x, pt.y - caliper.p2.y)
        if (d1 < 14) {
          setDraggedHandle('p1')
          return
        }
        if (d2 < 14) {
          setDraggedHandle('p2')
          return
        }
      }

      // Start new caliper measurement
      setCaliper({ p1: pt, p2: pt })
      setIsDrawingCaliper(true)
    } else if (activeTool === 'brush') {
      setCurrentPath([pt])
    }
  }

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    const pt = getNormalizedCoords(e)
    setMousePos(pt)

    if (activeTool === 'ruler') {
      if (draggedHandle && caliper) {
        if (draggedHandle === 'p1') {
          setCaliper({ ...caliper, p1: pt })
        } else {
          setCaliper({ ...caliper, p2: pt })
        }
      } else if (isDrawingCaliper && caliper) {
        setCaliper({ ...caliper, p2: pt })
      }
    } else if (activeTool === 'brush' && currentPath) {
      setCurrentPath(prev => (prev ? [...prev, pt] : [pt]))
    }
  }

  function handleMouseUp() {
    if (isDrawingCaliper) {
      setIsDrawingCaliper(false)
    }
    if (draggedHandle) {
      setDraggedHandle(null)
    }
    if (currentPath) {
      setBrushPaths(prev => [...prev, currentPath])
      setCurrentPath(null)
    }
  }

  const measuredDistanceMm = caliper ? calcDistanceMm(caliper.p1, caliper.p2) : 0

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
        onMouseLeave={() => {
          setMousePos(null)
          handleMouseUp()
        }}
      >
        {/* HUD Top Bar */}
        <div className="absolute top-2.5 inset-x-3 z-20 flex items-center justify-between pointer-events-none">
          <div className="font-mono text-[10px] bg-bg/85 border border-border2 px-2 py-0.5 text-text2">
            Axial · Slice {String(frameSlice).padStart(3, '0')} / {scan.sliceCount}
          </div>
          <div className="flex items-center gap-1.5">
            {caliper && (
              <div className="font-mono text-[10px] bg-teal/15 border border-teal/40 px-2 py-0.5 text-teal font-bold shadow-[0_0_8px_rgba(13,197,160,0.2)]">
                Caliper: {measuredDistanceMm.toFixed(1)} mm
              </div>
            )}
            <div className="font-mono text-[10px] bg-bg/85 border border-border2 px-2 py-0.5 text-teal">
              {zoomLevel}x
            </div>
          </div>
        </div>

        {/* Center Image Viewport Container */}
        <div className="relative flex-1 flex items-center justify-center p-3 overflow-hidden">
          {/* Viewport Crosshair Guides */}
          {activeTool === 'crosshair' && mousePos && (
            <div className="absolute inset-0 pointer-events-none z-10">
              <div
                className="absolute left-0 right-0 h-px bg-teal/40"
                style={{ top: `${(mousePos.y / 256) * 100}%` }}
              />
              <div
                className="absolute top-0 bottom-0 w-px bg-teal/40"
                style={{ left: `${(mousePos.x / 256) * 100}%` }}
              />
            </div>
          )}

          <div
            ref={imageContainerRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            className="relative aspect-square w-full max-w-[280px] xl:max-w-[320px] overflow-hidden rounded border border-border/70 bg-black shadow-2xl transition-transform duration-150 ease-out cursor-crosshair"
            style={{ transform: `scale(${zoomLevel})` }}
          >
            {/* Real MRI Slice Image with AI Segmentation Overlay */}
            <img
              src={frameSrc}
              onError={e => {
                const img = e.currentTarget
                if (!img.src.includes('p01-t1c-seg-slice')) {
                  img.src = `/demo-assets/p01-t1c-seg-slice-${frameSlice}.png`
                }
              }}
              alt={`Axial MRI scan slice ${frameSlice}`}
              className={cn(
                'h-full w-full object-contain pointer-events-none select-none',
                overlayMode === 'none' && 'grayscale contrast-125 brightness-110'
              )}
              draggable={false}
            />

            {/* SVG Annotation & Interactive Caliper Layer */}
            <svg
              viewBox="0 0 256 256"
              className="absolute inset-0 w-full h-full pointer-events-none"
            >
              {/* Brush Stored Strokes */}
              {brushPaths.map((path, idx) => (
                <polyline
                  key={idx}
                  points={path.map(p => `${p.x},${p.y}`).join(' ')}
                  fill="none"
                  stroke="#0DC5A0"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.8"
                />
              ))}

              {/* Current Active Brush Stroke */}
              {currentPath && currentPath.length > 1 && (
                <polyline
                  points={currentPath.map(p => `${p.x},${p.y}`).join(' ')}
                  fill="none"
                  stroke="#0DC5A0"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.9"
                />
              )}

              {/* Interactive Caliper Ruler */}
              {caliper && (
                <g className="cursor-pointer">
                  {/* Outer glow line */}
                  <line
                    x1={caliper.p1.x}
                    y1={caliper.p1.y}
                    x2={caliper.p2.x}
                    y2={caliper.p2.y}
                    stroke="rgba(13,197,160,0.3)"
                    strokeWidth="4"
                  />
                  {/* Dashed measurement line */}
                  <line
                    x1={caliper.p1.x}
                    y1={caliper.p1.y}
                    x2={caliper.p2.x}
                    y2={caliper.p2.y}
                    stroke="#0DC5A0"
                    strokeWidth="1.5"
                    strokeDasharray="4,3"
                  />
                  {/* Endpoint 1 Handle */}
                  <circle
                    cx={caliper.p1.x}
                    y1={caliper.p1.y}
                    cy={caliper.p1.y}
                    r="4"
                    fill="#0DC5A0"
                    stroke="#060810"
                    strokeWidth="1.5"
                    className="hover:scale-150 transition-transform"
                  />
                  {/* Endpoint 2 Handle */}
                  <circle
                    cx={caliper.p2.x}
                    cy={caliper.p2.y}
                    r="4"
                    fill="#0DC5A0"
                    stroke="#060810"
                    strokeWidth="1.5"
                    className="hover:scale-150 transition-transform"
                  />
                </g>
              )}
            </svg>

            {/* Floating Distance Badge above the caliper line */}
            {caliper && (
              <div
                className="absolute pointer-events-none font-mono text-[9px] font-bold text-teal bg-bg/95 border border-teal/40 px-1.5 py-0.5 rounded shadow-lg -translate-x-1/2 -translate-y-full"
                style={{
                  left: `${((caliper.p1.x + caliper.p2.x) / 2 / 256) * 100}%`,
                  top: `${((caliper.p1.y + caliper.p2.y) / 2 / 256) * 100 - 3}%`,
                }}
              >
                {measuredDistanceMm.toFixed(1)} mm
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
            onClick={() => setActiveTool('ruler')}
            title="Measure diameter (click & drag to measure)"
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
            onClick={() => setActiveTool('brush')}
            title="Brush annotation (click & drag to draw)"
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

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => {
              setCaliper(null)
              setBrushPaths([])
              setCurrentPath(null)
            }}
            title="Clear annotations"
            aria-label="Clear annotations"
            className="w-8 h-8 flex items-center justify-center bg-surface2 border border-border2 text-text3 hover:text-danger hover:border-danger transition-colors"
          >
            <Trash2 size={13} />
          </button>

          <button
            type="button"
            onClick={() => {
              setSlice(82)
              setZoomLevel(1)
              setOverlayMode('all')
              setCaliper({ p1: { x: 106, y: 118 }, p2: { x: 175, y: 118 } })
              setBrushPaths([])
              setActiveTool('ruler')
            }}
            title="Reset viewer"
            aria-label="Reset viewer"
            className="w-8 h-8 flex items-center justify-center bg-surface2 border border-border2 text-text3 hover:text-teal hover:border-teal transition-colors"
          >
            <RotateCcw size={13} />
          </button>
        </div>
      </div>
    </aside>
  )
}
