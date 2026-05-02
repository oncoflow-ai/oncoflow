import type { Patient, Scan, Summary } from '@/types'

export const mockPatients: Patient[] = [
  {
    id: 'P-9001',
    name: 'Demo Patient P01',
    dob: '1975-06-01',
    diagnosis: 'Demo lesion (sample BraTS volumes)',
    diagnosisLocation: 'See repo data/P01',
    assignedPhysicianId: 'DR-001',
    status: 'active',
    scanCount: 0,
    lastScanDate: '2024-01-01',
  },
  {
    id: 'P-1029',
    name: 'Sarah Jenkins',
    dob: '1994-07-22',
    diagnosis: 'Osteosarcoma',
    diagnosisLocation: 'Distal Left Femur',
    assignedPhysicianId: 'DR-001',
    status: 'active',
    scanCount: 3,
    lastScanDate: '2026-03-08',
  },
  {
    id: 'P-1031',
    name: 'David Levi',
    dob: '1958-11-03',
    diagnosis: 'Glioblastoma',
    diagnosisLocation: 'Right Temporal Lobe',
    assignedPhysicianId: 'DR-001',
    status: 'review',
    scanCount: 5,
    lastScanDate: '2026-03-10',
  },
  {
    id: 'P-1044',
    name: 'Miriam Cohen',
    dob: '1972-04-15',
    diagnosis: 'Breast Carcinoma Stage III',
    diagnosisLocation: 'Left Breast',
    assignedPhysicianId: 'DR-001',
    status: 'active',
    scanCount: 2,
    lastScanDate: '2026-02-20',
  },
  {
    id: 'P-1051',
    name: 'Jonathan Weiss',
    dob: '1965-09-28',
    diagnosis: 'Non-Hodgkin Lymphoma',
    diagnosisLocation: 'Mediastinal',
    assignedPhysicianId: 'DR-001',
    status: 'active',
    scanCount: 4,
    lastScanDate: '2026-03-05',
  },
  {
    id: 'P-1062',
    name: 'Noa Shapiro',
    dob: '1989-01-07',
    diagnosis: 'Renal Cell Carcinoma',
    diagnosisLocation: 'Right Kidney',
    assignedPhysicianId: 'DR-001',
    status: 'active',
    scanCount: 1,
    lastScanDate: '2026-03-01',
  },
  {
    id: 'P-1073',
    name: 'Yosef Mizrahi',
    dob: '1952-06-19',
    diagnosis: 'Colorectal Adenocarcinoma',
    diagnosisLocation: 'Sigmoid Colon',
    assignedPhysicianId: 'DR-001',
    status: 'active',
    scanCount: 6,
    lastScanDate: '2026-03-12',
  },
  {
    id: 'P-1081',
    name: 'Rachel Ben-David',
    dob: '1968-12-30',
    diagnosis: 'Pancreatic Ductal Adenocarcinoma',
    diagnosisLocation: 'Pancreatic Head',
    assignedPhysicianId: 'DR-001',
    status: 'review',
    scanCount: 2,
    lastScanDate: '2026-03-07',
  },
  {
    id: 'P-1094',
    name: 'Eitan Goldberg',
    dob: '1947-08-14',
    diagnosis: 'Lung Adenocarcinoma',
    diagnosisLocation: 'Left Lower Lobe',
    assignedPhysicianId: 'DR-001',
    status: 'active',
    scanCount: 3,
    lastScanDate: '2026-03-11',
  },
]

export const mockScans: Record<string, Scan[]> = {
  'P-9001': [],
  'P-1029': [
    { id: 'SCN-0039', patientId: 'P-1029', studyLabel: 'MRI Study #1', date: '2025-09-14', modality: 'MRI', sequence: 'T1W', plane: 'AXIAL', sliceCount: 128, resolution: '1.2mm iso', volumeMm3: 18400, maxDiameterMm: 34.2, isAnnotated: true },
    { id: 'SCN-0040', patientId: 'P-1029', studyLabel: 'MRI Study #2', date: '2025-12-02', modality: 'MRI', sequence: 'T1W', plane: 'AXIAL', sliceCount: 128, resolution: '1.2mm iso', volumeMm3: 15200, maxDiameterMm: 31.1, isAnnotated: true },
    { id: 'SCN-0041', patientId: 'P-1029', studyLabel: 'MRI Study #3', date: '2026-03-08', modality: 'MRI', sequence: 'T1W', plane: 'AXIAL', sliceCount: 128, resolution: '1.2mm iso', volumeMm3: 12480, maxDiameterMm: 28.4, isAnnotated: true },
  ],
  'P-1031': [
    { id: 'SCN-0020', patientId: 'P-1031', studyLabel: 'MRI Study #1', date: '2025-06-10', modality: 'MRI', sequence: 'T2-FLAIR', plane: 'AXIAL', sliceCount: 160, resolution: '1.0mm iso', volumeMm3: 22100, maxDiameterMm: 38.5, isAnnotated: true },
    { id: 'SCN-0021', patientId: 'P-1031', studyLabel: 'MRI Study #2', date: '2025-08-22', modality: 'MRI', sequence: 'T2-FLAIR', plane: 'AXIAL', sliceCount: 160, resolution: '1.0mm iso', volumeMm3: 24800, maxDiameterMm: 41.0, isAnnotated: true },
    { id: 'SCN-0022', patientId: 'P-1031', studyLabel: 'MRI Study #3', date: '2025-11-05', modality: 'MRI', sequence: 'T2-FLAIR', plane: 'AXIAL', sliceCount: 160, resolution: '1.0mm iso', volumeMm3: 27600, maxDiameterMm: 43.8, isAnnotated: true },
    { id: 'SCN-0023', patientId: 'P-1031', studyLabel: 'MRI Study #4', date: '2026-01-18', modality: 'MRI', sequence: 'T2-FLAIR', plane: 'AXIAL', sliceCount: 160, resolution: '1.0mm iso', volumeMm3: 31200, maxDiameterMm: 46.2, isAnnotated: true },
    { id: 'SCN-0024', patientId: 'P-1031', studyLabel: 'MRI Study #5', date: '2026-03-10', modality: 'MRI', sequence: 'T2-FLAIR', plane: 'AXIAL', sliceCount: 160, resolution: '1.0mm iso', volumeMm3: 34500, maxDiameterMm: 49.1, isAnnotated: false },
  ],
  'P-1044': [
    { id: 'SCN-0060', patientId: 'P-1044', studyLabel: 'MRI Study #1', date: '2025-11-10', modality: 'MRI', sequence: 'DCE', plane: 'AXIAL', sliceCount: 100, resolution: '1.5mm iso', volumeMm3: 9800, maxDiameterMm: 26.0, isAnnotated: true },
    { id: 'SCN-0061', patientId: 'P-1044', studyLabel: 'MRI Study #2', date: '2026-02-20', modality: 'MRI', sequence: 'DCE', plane: 'AXIAL', sliceCount: 100, resolution: '1.5mm iso', volumeMm3: 7200, maxDiameterMm: 22.5, isAnnotated: true },
  ],
  'P-1051': [
    { id: 'SCN-0080', patientId: 'P-1051', studyLabel: 'PET-CT Study #1', date: '2025-07-14', modality: 'PET-CT', sequence: 'FDG', plane: 'CORONAL', sliceCount: 120, resolution: '2.0mm iso', volumeMm3: 45000, maxDiameterMm: 55.0, isAnnotated: true },
    { id: 'SCN-0081', patientId: 'P-1051', studyLabel: 'PET-CT Study #2', date: '2025-10-22', modality: 'PET-CT', sequence: 'FDG', plane: 'CORONAL', sliceCount: 120, resolution: '2.0mm iso', volumeMm3: 42000, maxDiameterMm: 53.2, isAnnotated: true },
    { id: 'SCN-0082', patientId: 'P-1051', studyLabel: 'PET-CT Study #3', date: '2026-01-08', modality: 'PET-CT', sequence: 'FDG', plane: 'CORONAL', sliceCount: 120, resolution: '2.0mm iso', volumeMm3: 41500, maxDiameterMm: 52.8, isAnnotated: true },
    { id: 'SCN-0083', patientId: 'P-1051', studyLabel: 'PET-CT Study #4', date: '2026-03-05', modality: 'PET-CT', sequence: 'FDG', plane: 'CORONAL', sliceCount: 120, resolution: '2.0mm iso', volumeMm3: 40800, maxDiameterMm: 52.0, isAnnotated: true },
  ],
  'P-1062': [
    { id: 'SCN-0100', patientId: 'P-1062', studyLabel: 'CT Study #1', date: '2026-03-01', modality: 'CT', sequence: 'CECT', plane: 'AXIAL', sliceCount: 80, resolution: '1.0mm iso', volumeMm3: 38000, maxDiameterMm: 42.5, isAnnotated: false },
  ],
  'P-1073': [
    { id: 'SCN-0110', patientId: 'P-1073', studyLabel: 'CT Study #1', date: '2024-09-05', modality: 'CT', sequence: 'CECT', plane: 'AXIAL', sliceCount: 90, resolution: '1.25mm iso', volumeMm3: 62000, maxDiameterMm: 58.0, isAnnotated: true },
    { id: 'SCN-0111', patientId: 'P-1073', studyLabel: 'CT Study #2', date: '2024-12-12', modality: 'CT', sequence: 'CECT', plane: 'AXIAL', sliceCount: 90, resolution: '1.25mm iso', volumeMm3: 48000, maxDiameterMm: 50.4, isAnnotated: true },
    { id: 'SCN-0112', patientId: 'P-1073', studyLabel: 'CT Study #3', date: '2025-03-20', modality: 'CT', sequence: 'CECT', plane: 'AXIAL', sliceCount: 90, resolution: '1.25mm iso', volumeMm3: 36500, maxDiameterMm: 44.0, isAnnotated: true },
    { id: 'SCN-0113', patientId: 'P-1073', studyLabel: 'CT Study #4', date: '2025-07-08', modality: 'CT', sequence: 'CECT', plane: 'AXIAL', sliceCount: 90, resolution: '1.25mm iso', volumeMm3: 24800, maxDiameterMm: 37.5, isAnnotated: true },
    { id: 'SCN-0114', patientId: 'P-1073', studyLabel: 'CT Study #5', date: '2025-11-15', modality: 'CT', sequence: 'CECT', plane: 'AXIAL', sliceCount: 90, resolution: '1.25mm iso', volumeMm3: 16200, maxDiameterMm: 30.2, isAnnotated: true },
    { id: 'SCN-0115', patientId: 'P-1073', studyLabel: 'CT Study #6', date: '2026-03-12', modality: 'CT', sequence: 'CECT', plane: 'AXIAL', sliceCount: 90, resolution: '1.25mm iso', volumeMm3: 10400, maxDiameterMm: 24.8, isAnnotated: true },
  ],
  'P-1081': [
    { id: 'SCN-0120', patientId: 'P-1081', studyLabel: 'CT Study #1', date: '2025-10-18', modality: 'CT', sequence: 'CECT', plane: 'AXIAL', sliceCount: 110, resolution: '1.0mm iso', volumeMm3: 28500, maxDiameterMm: 38.0, isAnnotated: true },
    { id: 'SCN-0121', patientId: 'P-1081', studyLabel: 'CT Study #2', date: '2026-03-07', modality: 'CT', sequence: 'CECT', plane: 'AXIAL', sliceCount: 110, resolution: '1.0mm iso', volumeMm3: 33200, maxDiameterMm: 41.5, isAnnotated: true },
  ],
  'P-1094': [
    { id: 'SCN-0130', patientId: 'P-1094', studyLabel: 'CT Study #1', date: '2025-08-30', modality: 'CT', sequence: 'HRCT', plane: 'AXIAL', sliceCount: 120, resolution: '0.8mm iso', volumeMm3: 15800, maxDiameterMm: 31.2, isAnnotated: true },
    { id: 'SCN-0131', patientId: 'P-1094', studyLabel: 'CT Study #2', date: '2025-12-08', modality: 'CT', sequence: 'HRCT', plane: 'AXIAL', sliceCount: 120, resolution: '0.8mm iso', volumeMm3: 15200, maxDiameterMm: 30.8, isAnnotated: true },
    { id: 'SCN-0132', patientId: 'P-1094', studyLabel: 'CT Study #3', date: '2026-03-11', modality: 'CT', sequence: 'HRCT', plane: 'AXIAL', sliceCount: 120, resolution: '0.8mm iso', volumeMm3: 14900, maxDiameterMm: 30.5, isAnnotated: true },
  ],
}

export const mockSummaries: Record<string, Summary> = {
  'P-9001': {
    patientId: 'P-9001',
    generatedAt: '2026-05-02T12:00:00Z',
    model: 'Demo narrative',
    text: `This roster row pairs with **data/P01** sample volumes for live uploads. Sign in as Radiologist, select Demo Patient P01, upload baseline and follow-up NIfTI volumes plus masks per DEMO.md, then open Longitudinal Comparison. Mock scans above stay empty until you rely on demo MRI scaffolding separately.`,
    recommendations: [
      'Follow longitudinal metrics after radiologist uploads complete.',
      'Discuss treatment planning only with your clinical team.',
    ],
  },
  'P-1029': {
    patientId: 'P-1029',
    generatedAt: '2026-03-09T08:14:00Z',
    model: 'MedGemma 1.5 (RAG-augmented)',
    text: `Comparing the most recent study (MRI Study #3, 2026-03-08) with the baseline acquisition (MRI Study #1, 2025-09-14), the osteosarcoma of the distal left femur demonstrates a **sustained and clinically significant response to neoadjuvant chemotherapy**. Volumetric analysis by nnU-Net segmentation shows a 32.2% reduction in total tumor volume (18,400 → 12,480 mm³), with maximum diameter decreasing from 34.2 mm to 28.4 mm.

The signal intensity pattern on T1W sequences has evolved, with progressive central hypointensity consistent with **necrosis and fibrosis** — hallmarks of treatment response in high-grade osteosarcoma. No new satellite lesions or skip metastases are identified in the visualized field. Periosteal reaction appears stable. The adjacent growth plate architecture is preserved.

**Key finding:** Three consecutive scans demonstrate monotonic volume reduction, suggesting durable chemotherapeutic sensitivity. If this trajectory continues, the patient may be eligible for limb-salvage surgical resection at next assessment. Recommend correlation with alkaline phosphatase and LDH trends.`,
  },
  'P-1031': {
    patientId: 'P-1031',
    generatedAt: '2026-03-11T07:30:00Z',
    model: 'MedGemma 1.5 (RAG-augmented)',
    text: `Sequential T2-FLAIR imaging over 9 months reveals **progressive disease** in this GBM case. Tumor volume has increased 56.1% from baseline (22,100 → 34,500 mm³), with maximum diameter now approaching 50 mm. The enhancing tumor core has expanded rightward, encroaching on the posterior limb of the internal capsule.

**Critical observation:** MRI Study #5 (2026-03-10) shows new FLAIR signal extending into the corpus callosum — a "butterfly" pattern suggesting **contralateral infiltration**. Perilesional edema has increased significantly. Mass effect on the right lateral ventricle is more pronounced.

The patient is currently on temozolomide + bevacizumab following surgical debulking. The volumetric progression curve suggests pseudoprogression is unlikely given the consistent upward trajectory. **Urgent multidisciplinary review is recommended** to evaluate candidacy for re-irradiation or clinical trial enrollment. MGMT methylation status should be reviewed in context of current progression pattern.`,
  },
  'P-1044': {
    patientId: 'P-1044',
    generatedAt: '2026-02-21T09:00:00Z',
    model: 'MedGemma 1.5 (RAG-augmented)',
    text: `DCE-MRI comparison between baseline (2025-11-10) and follow-up (2026-02-20) demonstrates a **26.5% reduction in tumor volume** (9,800 → 7,200 mm³) with maximum diameter decreasing from 26.0 mm to 22.5 mm. Kinetic analysis shows decreased wash-in rate and reduced peak enhancement, consistent with reduced vascularity and **favorable neoadjuvant response**.

The lesion morphology has become more irregular with areas of central low signal on T2, suggestive of fibrous stromal replacement. No new axillary adenopathy is identified on the visualized field. Skin thickening has resolved.

Response assessment by RECIST 1.1 criteria: **Partial Response**. The patient is a candidate for surgical planning. Pre-operative MRI should include DWI sequences for complete characterization. Recommend breast surgery consultation.`,
  },
  'P-1051': {
    patientId: 'P-1051',
    generatedAt: '2026-03-06T10:15:00Z',
    model: 'MedGemma 1.5 (RAG-augmented)',
    text: `Four sequential PET-CT studies over 8 months show **stable disease with mild volumetric decline** in this mediastinal Non-Hodgkin Lymphoma. Total metabolically active volume has decreased from 45,000 to 40,800 mm³ (−9.3%), and maximum diameter has reduced from 55.0 to 52.0 mm. SUVmax has decreased on successive scans (data from PET metadata).

No new FDG-avid lesions are identified. The superior vena cava compression noted on baseline study has slightly improved with reduction in adjacent lymph node size. The residual mass demonstrates decreased tracer uptake, consistent with **treatment response per Deauville criteria** (score 3→2).

Current regimen (R-CHOP) appears to be achieving metabolic response. **Continue current protocol** with next assessment at 3-month interval. Consider end-of-treatment PET-CT to evaluate for consolidation radiotherapy candidacy.`,
  },
  'P-1062': {
    patientId: 'P-1062',
    generatedAt: '2026-03-02T08:45:00Z',
    model: 'MedGemma 1.5 (RAG-augmented)',
    text: `Baseline CT study (2026-03-01) documents a **38,000 mm³ right renal mass** with maximum diameter 42.5 mm. CECT characteristics include heterogeneous enhancement in the nephrographic phase with a washout ratio consistent with clear cell RCC phenotype. No evidence of venous thrombus extension into the renal vein or IVC on available sequences.

This is the patient's first imaging study in our system. **No comparative data available for volumetric trend analysis.** Regional lymph nodes appear within normal size limits. No radiologically identified distant metastases on the visualized thoracic and abdominal field.

Staging is cT1bN0M0 (provisional). The mass is confined to the kidney with preserved perinephric fat planes. **Recommend urologic oncology consultation** for surgical planning (partial vs. radical nephrectomy). Baseline functional nuclear medicine scan may be considered if partial nephrectomy is planned.`,
  },
  'P-1073': {
    patientId: 'P-1073',
    generatedAt: '2026-03-13T07:00:00Z',
    model: 'MedGemma 1.5 (RAG-augmented)',
    text: `Six-point longitudinal CT analysis over 18 months documents an **extraordinary response to FOLFOX + bevacizumab** in this sigmoid colorectal adenocarcinoma. Total tumor volume has decreased from 62,000 mm³ at baseline (2024-09-05) to 10,400 mm³ at most recent study (2026-03-12) — an **83.2% volumetric reduction**. Maximum diameter has decreased from 58.0 mm to 24.8 mm.

The reduction trajectory is consistent across all time points, with no evidence of plateau or inflection suggesting acquired resistance. The tumor now appears as a low-density fibrotic mass with no significant arterial phase enhancement — radiologically consistent with near-complete pathological response.

**Surgical conversion assessment:** Given the dramatic response, the previously unresectable sigmoid mass now appears technically resectable with potentially curative intent. The relationship to adjacent mesenteric vessels has normalized. **Colorectal surgery consultation is strongly recommended** for resection planning. Liver protocol CT shows no new hepatic metastases. CEA trend should be reviewed.`,
  },
  'P-1081': {
    patientId: 'P-1081',
    generatedAt: '2026-03-08T09:30:00Z',
    model: 'MedGemma 1.5 (RAG-augmented)',
    text: `Comparison between baseline (2025-10-18) and follow-up CT (2026-03-07) reveals **progressive disease** in this pancreatic ductal adenocarcinoma. Tumor volume has increased from 28,500 to 33,200 mm³ (+16.5%), with maximum diameter increasing from 38.0 to 41.5 mm over approximately 4.5 months.

The lesion remains centered in the pancreatic head with encasement of the superior mesenteric artery (>180° circumferential involvement on follow-up, upgraded from borderline on baseline). The superior mesenteric vein appears patent but with deformity. **Borderline resectable status has been downgraded to locally advanced unresectable.**

Biliary stent is in satisfactory position. No new hepatic lesions identified. Response to current gemcitabine/nab-paclitaxel regimen appears inadequate. **Urgent multidisciplinary oncology board review recommended** to evaluate FOLFIRINOX conversion therapy or clinical trial options. CA 19-9 trend correlation advised.`,
  },
  'P-1094': {
    patientId: 'P-1094',
    generatedAt: '2026-03-12T08:00:00Z',
    model: 'MedGemma 1.5 (RAG-augmented)',
    text: `Three CT studies spanning 6.5 months show **stable disease with minimal volumetric change** in this left lower lobe lung adenocarcinoma. Volume has decreased from 15,800 mm³ (baseline, 2025-08-30) to 14,900 mm³ (2026-03-11) — a modest **−5.7% reduction**. Maximum diameter has decreased marginally from 31.2 to 30.5 mm.

On HRCT, the lesion demonstrates a ground-glass opacity component with a solid core. The ratio of solid-to-GGO has remained stable, and no new satellite nodules are identified in the left lung. Mediastinal and hilar lymph nodes are within normal size limits. No pleural effusion.

**Assessment: Stable disease per RECIST 1.1.** The patient is receiving osimertinib (EGFR-targeted therapy, exon 19 deletion confirmed). The minimal volumetric decline over 6 months is consistent with typical osimertinib response kinetics in EGFR-mutant NSCLC. **Continue current regimen** with next surveillance CT in 3 months. Monitor for resistance patterns (T790M, C797S).`,
  },
}
