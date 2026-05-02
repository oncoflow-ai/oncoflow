export type PatientStatus = 'active' | 'review'

export interface Patient {
  id: string               // e.g. "P-1029"
  name: string
  dob: string              // ISO date "YYYY-MM-DD"
  diagnosis: string
  diagnosisLocation: string
  assignedPhysicianId: string
  status: PatientStatus
  scanCount: number
  lastScanDate: string     // ISO date
}

export interface Scan {
  id: string               // e.g. "SCN-0041"
  patientId: string
  studyLabel: string       // e.g. "MRI Study #3"
  date: string             // ISO date
  modality: string         // e.g. "MRI"
  sequence: string         // e.g. "T1W"
  plane: string            // e.g. "AXIAL"
  sliceCount: number
  resolution: string       // e.g. "1.2mm iso"
  volumeMm3: number
  maxDiameterMm: number
  isAnnotated: boolean
}

export interface Summary {
  patientId: string
  generatedAt: string      // ISO datetime
  model: string
  text: string
}

export interface MriUrl {
  url: string
  expiresAt: string        // ISO datetime
}

export interface Physician {
  id: string               // e.g. "DR-001"
  name: string
  initials: string
}

export type BackendJobStatus = 'queued' | 'running' | 'failed' | 'completed'

export interface BackendJobError {
  code: string
  message: string
  details?: Record<string, unknown> | null
}

export interface BackendJobSubmission {
  jobId: string
  studyId: string
  status: BackendJobStatus
  stage: string
  submittedAt: string
}

export interface BackendJobStatusResponse extends BackendJobSubmission {
  error: BackendJobError | null
}

export interface BackendArtifactRef {
  artifactKind: string
  storageRoot: string
  relativePath: string
}

export interface BackendBoundingBox3D {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  zMin: number
  zMax: number
}

export interface BackendLesionMeasurements {
  volumeMm3: number
  longestDiameterMm: number
}

export interface BackendLesionResult {
  lesionId: string
  boundingBox: BackendBoundingBox3D
  measurements: BackendLesionMeasurements
  maskArtifact: BackendArtifactRef
  reviewArtifacts: BackendArtifactRef[]
  metadata?: Record<string, unknown> | null
}

export interface BackendCaseResult {
  studyId: string
  resultArtifact: BackendArtifactRef
  lesions: BackendLesionResult[]
  needsReview: boolean
  caseQcReasons: string[]
}

export interface BackendStudyListItem {
  studyId: string
  sourceKind: string
  sourceLabel?: string | null
  acquiredAt?: string | null
  createdAt: string
  jobStatus: string
  hasResults: boolean
}

export interface BackendComparisonMetrics {
  volumeACm3: number
  volumeBCm3: number
  deltaCm3: number
  pctChange: number
  diceOverlap?: number | null
  hd95Mm?: number | null
  recistAMm?: number | null
  recistBMm?: number | null
  recistRatio?: number | null
  growthRateCm3PerDay?: number | null
  registrationNcc?: number | null
  volDeltaCiHalfCm3?: number | null
  method?: string | null
  backend?: string | null
  didResegment?: boolean | null
}

export interface BackendComparisonResponse {
  comparisonId: string
  baselineStudyId: string
  followupStudyId: string
  baselineAcquiredAt?: string | null
  followupAcquiredAt?: string | null
  metrics: BackendComparisonMetrics
  interpretation?: string | null
  notes: string[]
  outputRelativePath: string
}
