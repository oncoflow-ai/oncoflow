import type { MriUrl } from '@/types'
import { delay } from './client'

export async function getMriUrl(scanId: string): Promise<MriUrl> {
  await delay(200)
  // Mock: returns a placeholder URL. Backend will return a real presigned S3 URL.
  return {
    url: `https://mock-dicom.oncoflow.internal/${scanId}.nii.gz`,
    expiresAt: new Date(Date.now() + 3600_000).toISOString(),
  }
}
