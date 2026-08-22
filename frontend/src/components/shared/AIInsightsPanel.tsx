import type { Summary } from '@/types'
import { formatDate } from '@/lib/utils'
import { Sparkles, Bot, Database, ShieldCheck, CheckCircle2 } from 'lucide-react'

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
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4 pb-3 border-b border-border2">
        <div className="flex items-center gap-2">
          <Sparkles size={14} className="text-teal" />
          <span className="font-mono text-[11px] font-bold tracking-widest uppercase text-teal">
            AI Multi-Agent Clinical Narrative
          </span>
          <span className="font-mono text-[10px] text-text3 ml-1">
            · {summary.model} · {formatDate(summary.generatedAt)}
          </span>
        </div>

        <div className="flex items-center gap-2 font-mono text-[10px]">
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-teal/10 text-teal border border-teal/20">
            <Bot size={11} /> Image Stream
          </span>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Database size={11} /> Prior Summary RAG
          </span>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <ShieldCheck size={11} /> Validated
          </span>
        </div>
      </div>

      <div className="font-sans text-[13px] text-text2 leading-[1.8] space-y-3">
        {renderBoldText(summary.text)}
      </div>

      {summary.recommendations && summary.recommendations.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border2">
          <div className="font-mono text-[11px] font-bold tracking-widest uppercase text-text1 mb-2">
            Actionable Recommendations
          </div>
          <ul className="space-y-1.5 font-sans text-[12px] text-text2">
            {summary.recommendations.map((rec, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <CheckCircle2 size={13} className="text-teal shrink-0 mt-0.5" />
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

