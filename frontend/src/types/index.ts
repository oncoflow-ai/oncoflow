export type PatientStatus = 'active' | 'review'

export type UserRole = 'admin' | 'doctor' | 'radiologist' | 'clinician' | 'patient'

/** Authenticated session user (mock login until backend JWT exists). */
export interface AuthenticatedUser {
  id: string
  name: string
  initials: string
  email?: string
  role: UserRole
  /** When role is patient, which roster record this portal shows */
  patientRecordId?: string
}

/** @deprecated Use AuthenticatedUser */
export type Physician = AuthenticatedUser

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
  /** Backend study UUIDs for demo longitudinal filtering */
  linkedStudyIds?: string[]
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
  recommendations?: string[]
}

export interface MriUrl {
  url: string
  expiresAt: string        // ISO datetime
}

export type BackendJobStatus = 'queued' | 'running' | 'failed' | 'completed'

/** Stored report metadata (mock persistence). */
export interface ClinicalReportEntry {
  id: string
  patientId: string
  title: string
  generatedAt: string
  summarySnippet: string
}

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
  [key: string]: number
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
  metadata?: Record<string, unknown> | null
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

export interface AppUser extends AuthenticatedUser {
  email: string
  password: string
}
