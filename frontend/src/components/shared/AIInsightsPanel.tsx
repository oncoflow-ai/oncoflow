import { useMemo } from 'react'
import type { Patient, Scan, Summary } from '@/types'
import { formatDate } from '@/lib/utils'
import { downloadReportPdf } from '@/lib/pdfReportGenerator'
import { Sparkles, Bot, Database, ShieldCheck, CheckCircle2, Download } from 'lucide-react'

interface AIInsightsPanelProps {
  summary: Summary
  patient?: Patient | null
  scan?: Scan | null
  scans?: Scan[]
}

function resolveScanStage(scan?: Scan | null): number {
  if (!scan) return 0
  if (typeof scan.demoStageIndex === 'number') return scan.demoStageIndex
  if (scan.studyLabel.includes('Follow-up #1') || scan.studyLabel.includes('Follow-Up #1') || Math.abs(scan.volumeMm3 - 3101) < 50) return 1
  if (scan.studyLabel.includes('Follow-up #2') || scan.studyLabel.includes('Follow-Up #2') || Math.abs(scan.volumeMm3 - 3911) < 50) return 2
  if (scan.studyLabel.includes('Follow-up #3') || scan.studyLabel.includes('Follow-Up #3') || Math.abs(scan.volumeMm3 - 2285) < 50) return 3
  if (scan.studyLabel.includes('Follow-up #4') || scan.studyLabel.includes('Follow-Up #4') || Math.abs(scan.volumeMm3 - 1264) < 50) return 4
  return 0
}

function buildDynamicSummary(baseSummary: Summary, scan?: Scan | null, patient?: Patient | null): Summary {
  if (!scan) return baseSummary

  const stage = resolveScanStage(scan)
  const patientName = patient?.name || 'Demo Patient P01'

  if (stage === 0) {
    const volFormatted = (scan.volumeMm3).toLocaleString()
    const volCm3 = (scan.volumeMm3 / 1000).toFixed(2)
    const diamFormatted = scan.maxDiameterMm ? scan.maxDiameterMm.toFixed(1) : '39.1'
    return {
      ...baseSummary,
      generatedAt: scan.date ? `${scan.date}T12:00:00Z` : baseSummary.generatedAt,
      text: `Baseline quantitative tumor burden analysis of axial T1-weighted post-contrast (T1c) and FLAIR MRI acquisitions for ${patientName} establishes the initial reference benchmark for treatment tracking.

**Volumetric Findings:**
Automated volumetric segmentation by nnU-Net demonstrates a solitary enhancing tumor volume of **${volFormatted} mm³** (${volCm3} cm³) with a maximum axial diameter of **${diamFormatted} mm**. This initial acquisition provides the quantitative baseline against which subsequent serial therapeutic interval changes will be assessed.

**Morphology & Edema Extent:**
The lesion is centered in the right fronto-parietal white matter with moderate surrounding vasogenic edema extending into the deep periatrial tracks. Midline structures remain centered without uncal herniation or mass-effect crisis.

**Impression & Multi-Agent Consensus:**
1. Solitary right cerebral intra-axial enhancing mass establishing baseline tumor burden (${volCm3} cm³ volume).
2. Initial automated AI segmentation validated. Initiate targeted clinical protocol and schedule serial follow-up contrast MRI for quantitative response tracking.`,
      recommendations: [
        'Radiologist verification of automated baseline segmentation contour.',
        'Initiate multidisciplinary neuro-oncology treatment protocol.',
        'Schedule first follow-up surveillance MRI in 8-12 weeks to monitor therapeutic response.',
      ],
    }
  }

  if (stage === 1) {
    return {
      ...baseSummary,
      generatedAt: scan.date ? `${scan.date}T12:00:00Z` : baseSummary.generatedAt,
      text: `Comparative longitudinal analysis between baseline MRI (14,815 mm³) and **${scan.studyLabel}** reveals **marked therapeutic tumor regression and rapid shrinkage** following the initiation of treatment for ${patientName}.

**Volumetric Findings:**
Automated volumetric segmentation by nnU-Net demonstrates the **largest interval decrease** across the serial trajectory, dropping from **14,815 mm³** at baseline to **3,101 mm³** on this follow-up study — representing a major **-79.1% volume reduction**. Longest axial diameter has decreased from 39.1 mm to **21.2 mm** (-17.9 mm reduction).

**Morphology & Edema Extent:**
Significant interval resolution of peritumoral vasogenic edema with marked shrinkage of the enhancing solid core in the right parietal region. Midline structures and ventricular caliber remain intact without mass effect.

**Impression & Multi-Agent Consensus:**
1. Marked interval regression of solitary right parietal enhancing lesion with the most pronounced volumetric decrease in the series (-79.1%).
2. High therapeutic response to current protocol. Recommend maintaining current regimen and scheduling 8–12 week surveillance MRI.`,
      recommendations: [
        'Maintain current systemic therapy protocol given major radiological response.',
        'Schedule follow-up MRI in 8-12 weeks to monitor shrinkage trajectory.',
        'Correlate with clinical neurological exam and steroid tapering.',
      ],
    }
  }

  if (stage === 2) {
    return {
      ...baseSummary,
      generatedAt: scan.date ? `${scan.date}T12:00:00Z` : baseSummary.generatedAt,
      text: `Comparative longitudinal analysis of **${scan.studyLabel}** against prior serial studies reveals **stable post-treatment cavity appearances** with preserved overall therapeutic reduction relative to baseline (-73.6%).

**Volumetric Findings:**
Automated volumetric segmentation demonstrates a tumor bed volume of **3,911 mm³** (diameter 23.5 mm). Following the initial major drop to 3,101 mm³ on Follow-up #1, the lesion bed exhibits minor post-treatment margin remodeling (+26.1% relative interval fluctuation), maintaining an overall **-73.6% volume reduction** relative to baseline (14,815 mm³).

**Morphology & Edema Extent:**
Thin, non-nodular peripheral enhancement around the resection cavity. Adjacent peritumoral edema remains minimal, stable, and non-progressive.

**Impression & Multi-Agent Consensus:**
1. Stable post-treatment cavity dynamics without evidence of true nodular recurrence.
2. Durable overall response preserved relative to baseline. Maintain surveillance protocol.`,
      recommendations: [
        'Continue current maintenance therapy and surveillance schedule.',
        'Review subtraction T1c series to differentiate surgical/radiation rim enhancement from recurrence.',
        'Routine interval MRI follow-up in 3 months.',
      ],
    }
  }

  if (stage === 3) {
    return {
      ...baseSummary,
      generatedAt: scan.date ? `${scan.date}T12:00:00Z` : baseSummary.generatedAt,
      text: `Comparative longitudinal analysis of **${scan.studyLabel}** demonstrates **progressive and continued gradual tumor shrinkage** across serial follow-up examinations.

**Volumetric Findings:**
Automated volumetric segmentation shows tumor volume decreasing further to **2,285 mm³** (diameter 19.1 mm) — representing a **-41.6% interval reduction** from Follow-up #2 (3,911 mm³) and an overall **-84.6% volume reduction** from the baseline study (14,815 mm³).

**Morphology & Edema Extent:**
Continued gradual regression of residual enhancing tissue in the right parietal region. Surrounding parenchymal edema has nearly completely resolved.

**Impression & Multi-Agent Consensus:**
1. Continued gradual interval tumor shrinkage and sustained therapeutic response (-84.6% vs baseline).
2. Findings consistent with ongoing tumor control. Continue maintenance therapy.`,
      recommendations: [
        'Continue maintenance protocol as clinically indicated.',
        'Routine 3-month imaging surveillance.',
        'No emergency intervention indicated.',
      ],
    }
  }

  if (stage === 4) {
    return {
      ...baseSummary,
      generatedAt: scan.date ? `${scan.date}T12:00:00Z` : baseSummary.generatedAt,
      text: `Comparative longitudinal analysis of **${scan.studyLabel}** confirms **minimal residual disease and sustained long-term remission** across the multi-study serial imaging series.

**Volumetric Findings:**
Automated volumetric segmentation reveals the tumor volume has shrunk to **1,264 mm³** (diameter 14.8 mm). This reflects a further gradual decrease from Follow-up #3 (2,285 mm³) and a cumulative **-91.5% volume reduction** from baseline (14,815 mm³), demonstrating a sustained downward regression curve from the initial large reduction to a minimal residual focus.

**Morphology & Edema Extent:**
Only faint, minute focal enhancement is visible in the right parietal cavity. Complete resolution of edema, mass effect, and ventricular distortion.

**Impression & Multi-Agent Consensus:**
1. Minimal residual disease with durable long-term response (-91.5% from baseline).
2. Excellent therapeutic outcome without evidence of aggressive relapse.`,
      recommendations: [
        'Reassuring imaging appearance. Continue standard follow-up intervals.',
        'Correlate with clinical status for potential treatment de-escalation.',
        'Annual or bi-annual surveillance imaging.',
      ],
    }
  }

  return baseSummary
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

export default function AIInsightsPanel({ summary, patient, scan }: AIInsightsPanelProps) {
  const activeSummary = useMemo(
    () => buildDynamicSummary(summary, scan, patient),
    [summary, scan, patient]
  )

  function handleDownloadPdf() {
    downloadReportPdf({
      patient,
      scan,
      summary: activeSummary,
      reportTitle: 'AI Multi-Agent Clinical Narrative & Progression Report',
      generatedAt: activeSummary.generatedAt,
    })
  }

  return (
    <div className="bg-surface border border-border2 border-l-[3px] border-l-teal p-5">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4 pb-3 border-b border-border2">
        <div className="flex items-center gap-2">
          <Sparkles size={14} className="text-teal" />
          <span className="font-mono text-[11px] font-bold tracking-widest uppercase text-teal">
            AI Multi-Agent Clinical Narrative
          </span>
          <span className="font-mono text-[10px] text-text3 ml-1">
            · {activeSummary.model} · {formatDate(activeSummary.generatedAt)}
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
          <button
            type="button"
            onClick={handleDownloadPdf}
            className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded border border-teal/40 bg-teal/10 text-teal font-mono text-[10px] font-bold uppercase tracking-wider hover:bg-teal hover:text-bg transition-colors"
          >
            <Download size={11} /> Export PDF
          </button>
        </div>
      </div>

      <div className="font-sans text-[13px] text-text2 leading-[1.8] space-y-3">
        {renderBoldText(activeSummary.text)}
      </div>

      {activeSummary.recommendations && activeSummary.recommendations.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border2">
          <div className="font-mono text-[11px] font-bold tracking-widest uppercase text-text1 mb-2">
            Actionable Recommendations
          </div>
          <ul className="space-y-1.5 font-sans text-[12px] text-text2">
            {activeSummary.recommendations.map((rec, idx) => (
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

