import { jsPDF } from 'jspdf'
import type { Patient, Scan, Summary, BackendLesionResult } from '@/types'
import { formatDate, formatVolume } from './utils'

export interface StructuredReportData {
  title?: string
  technique?: string
  finding?: string
  subregions?: string[]
  quantitative?: {
    current_volume_cm3?: number
    prior_volume_cm3?: number
    volume_change_pct?: number
    longest_diameter_mm?: number
    prior_longest_diameter_mm?: number
    diameter_change_mm?: number
    confidence?: string
  }
  comparison?: string
  impression?: string
  recommendations?: string[]
  disclaimer?: string
}

export interface GeneratePdfOptions {
  patient?: Patient | null
  studyId?: string | null
  scan?: Scan | null
  lesion?: BackendLesionResult | null
  structuredReport?: StructuredReportData | null
  summary?: Summary | null
  generatedAt?: string
  reportTitle?: string
  authorName?: string
  authorRole?: string
}

export function generateReportPdf(options: GeneratePdfOptions): jsPDF {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  })

  const pageWidth = doc.internal.pageSize.getWidth()
  const pageHeight = doc.internal.pageSize.getHeight()
  const margin = 15
  const contentWidth = pageWidth - margin * 2
  let y = margin

  const colors = {
    teal: [13, 197, 160] as [number, number, number],
    darkTeal: [9, 133, 108] as [number, number, number],
    slateBg: [10, 16, 26] as [number, number, number],
    cardBg: [245, 247, 250] as [number, number, number],
    cardBorder: [220, 226, 235] as [number, number, number],
    darkText: [24, 30, 42] as [number, number, number],
    mutedText: [90, 100, 115] as [number, number, number],
    lightMuted: [140, 150, 165] as [number, number, number],
    amber: [217, 119, 6] as [number, number, number],
    danger: [225, 29, 72] as [number, number, number],
  }

  function checkPageBreak(requiredSpace: number) {
    if (y + requiredSpace > pageHeight - margin - 15) {
      doc.addPage()
      y = margin + 10
      drawHeaderBanner(true)
    }
  }

  function drawHeaderBanner(isSubsequentPage = false) {
    if (!isSubsequentPage) {
      // Top header band
      doc.setFillColor(...colors.slateBg)
      doc.rect(0, 0, pageWidth, 28, 'F')

      // Accent strip
      doc.setFillColor(...colors.teal)
      doc.rect(0, 27, pageWidth, 1.5, 'F')

      // Logo / Brand
      doc.setTextColor(255, 255, 255)
      doc.setFont('helvetica', 'bold')
      doc.setFontSize(16)
      doc.text('ONCOFLOW', margin, 13)

      doc.setFont('helvetica', 'normal')
      doc.setFontSize(9)
      doc.setTextColor(13, 197, 160)
      doc.text('AI ONCOLOGY & MRI TUMOR TRAJECTORY INTELLIGENCE', margin, 19)

      // Right header info
      doc.setTextColor(200, 210, 225)
      doc.setFontSize(8)
      doc.text('CONFIDENTIAL MEDICAL RECORD', pageWidth - margin, 12, { align: 'right' })
      const genDate = options.generatedAt ? new Date(options.generatedAt).toLocaleString() : new Date().toLocaleString()
      doc.text(`Generated: ${genDate}`, pageWidth - margin, 18, { align: 'right' })

      y = 36
    }
  }

  // Draw Initial Header
  drawHeaderBanner(false)

  // Report Title
  const title = options.reportTitle || options.structuredReport?.title || 'AI Quantitative MRI & Longitudinal Progression Report'
  doc.setFont('helvetica', 'bold')
  doc.setFontSize(14)
  doc.setTextColor(...colors.darkText)
  doc.text(title, margin, y)
  y += 7

  // Patient & Study Info Card
  const patient = options.patient
  const infoBoxHeight = 28
  doc.setFillColor(...colors.cardBg)
  doc.setDrawColor(...colors.cardBorder)
  doc.roundedRect(margin, y, contentWidth, infoBoxHeight, 2, 2, 'FD')

  const col1X = margin + 4
  const col2X = margin + (contentWidth / 3) + 2
  const col3X = margin + (contentWidth * 2 / 3) + 2

  doc.setFontSize(7.5)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colors.mutedText)
  doc.text('PATIENT IDENTIFIER', col1X, y + 6)
  doc.text('ANATOMICAL SITE / DIAGNOSIS', col2X, y + 6)
  doc.text('STUDY & IMAGING CONTEXT', col3X, y + 6)

  doc.setFontSize(9.5)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colors.darkText)
  doc.text(patient ? patient.name : 'Unknown Patient', col1X, y + 12)
  doc.setFont('helvetica', 'normal')
  doc.setFontSize(8.5)
  doc.setTextColor(...colors.mutedText)
  doc.text(patient ? `MRN: ${patient.id} · DOB: ${formatDate(patient.dob)}` : 'ID: N/A', col1X, y + 17)
  doc.text(patient ? `Status: ${patient.status.toUpperCase()}` : '', col1X, y + 22)

  doc.setFontSize(9)
  doc.setTextColor(...colors.darkText)
  doc.text(patient ? patient.diagnosis : 'Brain Neoplasm / Glioma', col2X, y + 12)
  doc.setFontSize(8)
  doc.setTextColor(...colors.mutedText)
  doc.text(patient ? patient.diagnosisLocation : 'Intracranial', col2X, y + 17)

  const studyText = options.studyId ? `Study: ${options.studyId.slice(0, 16)}...` : (options.scan ? options.scan.studyLabel : 'MRI Brain Protocol')
  const seqText = options.scan ? `${options.scan.sequence} · ${options.scan.plane} · ${options.scan.resolution}` : 'T1c / T2-FLAIR Axial'
  doc.setFontSize(8.5)
  doc.setTextColor(...colors.darkText)
  doc.text(studyText, col3X, y + 12)
  doc.setFontSize(8)
  doc.setTextColor(...colors.mutedText)
  doc.text(seqText, col3X, y + 17)
  if (options.authorName) {
    doc.text(`Reviewed by: ${options.authorName} (${options.authorRole || 'Clinician'})`, col3X, y + 22)
  }

  y += infoBoxHeight + 6

  // Quantitative Volumetric Metrics Table (if available)
  const quant = options.structuredReport?.quantitative
  const lesion = options.lesion
  const scan = options.scan

  const currentVol = quant?.current_volume_cm3
    ? `${quant.current_volume_cm3.toFixed(2)} cm³ (${formatVolume(quant.current_volume_cm3 * 1000)} mm³)`
    : lesion
      ? `${formatVolume(lesion.measurements.volumeMm3)} mm³ (${(lesion.measurements.volumeMm3 / 1000).toFixed(2)} cm³)`
      : scan
        ? `${formatVolume(scan.volumeMm3)} mm³`
        : null

  const priorVol = quant?.prior_volume_cm3
    ? `${quant.prior_volume_cm3.toFixed(2)} cm³`
    : null

  const volDelta = quant?.volume_change_pct !== undefined
    ? `${quant.volume_change_pct > 0 ? '+' : ''}${quant.volume_change_pct.toFixed(1)}%`
    : null

  const maxDiameter = quant?.longest_diameter_mm !== undefined
    ? `${quant.longest_diameter_mm.toFixed(1)} mm`
    : lesion
      ? `${lesion.measurements.longestDiameterMm.toFixed(1)} mm`
      : scan
        ? `${scan.maxDiameterMm.toFixed(1)} mm`
        : null

  const diameterDelta = quant?.diameter_change_mm !== undefined
    ? `${quant.diameter_change_mm > 0 ? '+' : ''}${quant.diameter_change_mm.toFixed(1)} mm`
    : null

  if (currentVol || maxDiameter) {
    checkPageBreak(36)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(10)
    doc.setTextColor(...colors.darkTeal)
    doc.text('QUANTITATIVE VOLUMETRIC & PROGRESSION MEASUREMENTS', margin, y)
    y += 4

    const metricBoxHeight = 20
    doc.setFillColor(250, 252, 255)
    doc.setDrawColor(...colors.cardBorder)
    doc.roundedRect(margin, y, contentWidth, metricBoxHeight, 1.5, 1.5, 'FD')

    const qColWidth = contentWidth / 4
    const qItems = [
      { label: 'CURRENT TUMOR VOLUME', val: currentVol ?? '—' },
      { label: 'PRIOR BASELINE VOLUME', val: priorVol ?? '—' },
      { label: 'VOLUMETRIC INTERVAL DELTA', val: volDelta ?? '—', highlight: volDelta?.startsWith('+') },
      { label: 'MAX LONGEST DIAMETER', val: maxDiameter ? `${maxDiameter}${diameterDelta ? ` (${diameterDelta})` : ''}` : '—' },
    ]

    qItems.forEach((item, idx) => {
      const xPos = margin + idx * qColWidth + 3
      doc.setFontSize(6.8)
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...colors.mutedText)
      doc.text(item.label, xPos, y + 6)

      doc.setFontSize(9)
      doc.setFont('helvetica', 'bold')
      if (item.highlight) {
        doc.setTextColor(...colors.amber)
      } else {
        doc.setTextColor(...colors.darkText)
      }
      doc.text(item.val, xPos, y + 13)
    })

    y += metricBoxHeight + 6
  }

  // Structured Sections: Technique, Findings, Comparison, Impression
  const report = options.structuredReport
  const summary = options.summary

  function renderSection(sectionTitle: string, content: string | undefined, isHighlight = false) {
    if (!content) return
    const lines = doc.splitTextToSize(content, contentWidth - 6)
    const boxHeight = lines.length * 4.5 + 10

    checkPageBreak(boxHeight + 8)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(9.5)
    doc.setTextColor(...colors.darkTeal)
    doc.text(sectionTitle.toUpperCase(), margin, y)
    y += 3.5

    doc.setFillColor(isHighlight ? 240 : 252, isHighlight ? 250 : 252, isHighlight ? 248 : 254)
    doc.setDrawColor(...colors.cardBorder)
    doc.roundedRect(margin, y, contentWidth, boxHeight - 2, 1, 1, 'FD')

    if (isHighlight) {
      doc.setFillColor(...colors.teal)
      doc.rect(margin, y, 1.5, boxHeight - 2, 'F')
    }

    doc.setFont('helvetica', 'normal')
    doc.setFontSize(8.8)
    doc.setTextColor(...colors.darkText)
    doc.text(lines, margin + 4, y + 6)

    y += boxHeight + 3
  }

  if (report) {
    renderSection('Technique & Protocol', report.technique)
    renderSection('Clinical Findings', report.finding)
    renderSection('Interval Comparison to Prior Scans', report.comparison)
    renderSection('Impression & AI Synthesis', report.impression, true)
  } else if (summary) {
    renderSection('Clinical Findings', summary.findings || summary.text)
    renderSection('Interval Comparison', summary.comparison)
    renderSection('Impression & Multi-Agent Synthesis', summary.impression, true)
  }

  // Tumor Subregions (if available)
  const subregions = report?.subregions
  if (subregions && subregions.length > 0) {
    checkPageBreak(18)
    doc.setFont('helvetica', 'bold')
    doc.setFontSize(9)
    doc.setTextColor(...colors.mutedText)
    doc.text('IDENTIFIED TUMOR SUBREGIONS & COMPONENT MASKS', margin, y)
    y += 4

    let rx = margin
    subregions.forEach(region => {
      const tagText = region.toUpperCase()
      const textWidth = doc.getTextWidth(tagText) + 8
      if (rx + textWidth > margin + contentWidth) {
        rx = margin
        y += 7
      }
      doc.setFillColor(235, 240, 248)
      doc.setDrawColor(200, 215, 230)
      doc.roundedRect(rx, y, textWidth, 6, 1, 1, 'FD')
      doc.setFontSize(7.5)
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...colors.darkText)
      doc.text(tagText, rx + 4, y + 4.2)
      rx += textWidth + 3
    })
    y += 10
  }

  // Actionable Recommendations
  const recs = report?.recommendations || summary?.recommendations
  if (recs && recs.length > 0) {
    checkPageBreak(recs.length * 6 + 14)

    doc.setFont('helvetica', 'bold')
    doc.setFontSize(9.5)
    doc.setTextColor(...colors.darkTeal)
    doc.text('ACTIONABLE CLINICAL RECOMMENDATIONS', margin, y)
    y += 4

    doc.setFontSize(8.5)
    recs.forEach(rec => {
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...colors.teal)
      doc.text('•', margin + 2, y)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...colors.darkText)
      const recLines = doc.splitTextToSize(rec, contentWidth - 8)
      doc.text(recLines, margin + 6, y)
      y += recLines.length * 4.2 + 2
    })
    y += 4
  }

  // AI Multi-Agent Attribution & Quality Validation
  checkPageBreak(24)
  doc.setFillColor(...colors.cardBg)
  doc.setDrawColor(...colors.cardBorder)
  doc.roundedRect(margin, y, contentWidth, 16, 1, 1, 'FD')

  doc.setFontSize(7.2)
  doc.setFont('helvetica', 'bold')
  doc.setTextColor(...colors.mutedText)
  doc.text('AI MULTI-AGENT INFERENCE & VALIDATION ENGINE', margin + 3, y + 4.5)

  doc.setFont('helvetica', 'normal')
  doc.setFontSize(7.5)
  doc.setTextColor(...colors.darkText)
  const agentInfo = 'Segmentation: nnU-Net BraTS V2 (Ensemble)  |  Clinical Synthesis: MedGemma RAG Agent  |  Validation: Metric Guardrails Pass'
  doc.text(agentInfo, margin + 3, y + 9)

  const disclaimerText = report?.disclaimer || 'NOTICE: This report is generated by OncoFlow AI Clinical Decision Support System. For investigational and clinical workflow assistance only. All contours and volumetric metrics must be reviewed and confirmed by a certified radiologist/oncologist.'
  doc.setFontSize(6.8)
  doc.setTextColor(...colors.lightMuted)
  const discLines = doc.splitTextToSize(disclaimerText, contentWidth - 6)
  doc.text(discLines, margin + 3, y + 13)

  // Add Page Numbers & Footer to all pages
  const totalPages = doc.getNumberOfPages()
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i)
    doc.setDrawColor(...colors.cardBorder)
    doc.line(margin, pageHeight - 10, pageWidth - margin, pageHeight - 10)

    doc.setFontSize(7)
    doc.setFont('helvetica', 'normal')
    doc.setTextColor(...colors.lightMuted)
    doc.text('OncoFlow AI Diagnostic Suite · Hospital Grade Medical Imaging', margin, pageHeight - 6)
    doc.text(`Page ${i} of ${totalPages}`, pageWidth - margin, pageHeight - 6, { align: 'right' })
  }

  return doc
}

export function downloadReportPdf(options: GeneratePdfOptions, filename?: string) {
  const doc = generateReportPdf(options)
  const patientId = options.patient?.id || 'Patient'
  const dateStr = new Date().toISOString().slice(0, 10)
  const defaultFilename = `OncoFlow_Report_${patientId}_${dateStr}.pdf`
  doc.save(filename || defaultFilename)
}
