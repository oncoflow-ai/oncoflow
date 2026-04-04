import type { Summary } from '@/types'
import { formatDate } from '@/lib/utils'
import { Sparkles } from 'lucide-react'

interface AIInsightsPanelProps {
  summary: Summary
}

function renderBoldText(text: string) {
  return text.split('\n\n').map((para, pi) => (
    <p key={pi} className="mb-3 last:mb-0">
      {para.split('**').map((part, i) =>
        i % 2 === 1
          ? <strong key={i} className="text-text1 font-semibold">{part}</strong>
          : part
      )}
    </p>
  ))
}

export default function AIInsightsPanel({ summary }: AIInsightsPanelProps) {
  return (
    <div className="bg-surface border border-border2 border-l-[3px] border-l-teal p-5">
      <div className="flex items-center gap-2 mb-3">
        <Sparkles size={14} className="text-teal" />
        <span className="font-mono text-[11px] font-bold tracking-widest uppercase text-teal">
          AI Clinical Narrative
        </span>
        <span className="font-mono text-[10px] text-text3 ml-1">
          · {summary.model} · Generated {formatDate(summary.generatedAt)}
        </span>
      </div>
      <div className="font-sans text-[13px] text-text2 leading-[1.8]">
        {renderBoldText(summary.text)}
      </div>
    </div>
  )
}
