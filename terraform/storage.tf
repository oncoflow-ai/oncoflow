# Google Cloud Storage Buckets
resource "google_storage_bucket" "nifti_bucket" {
  name          = "${var.project_id}-nifti"
  location      = var.region
  force_destroy = true
  
  uniform_bucket_level_access = true

  depends_on = [google_project_service.required_apis]
}

resource "google_storage_bucket" "masks_bucket" {
  name          = "${var.project_id}-masks"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  depends_on = [google_project_service.required_apis]
}

resource "google_storage_bucket" "reports_bucket" {
  name          = "${var.project_id}-reports"
  location      = var.region
  force_destroy = true

  uniform_bucket_level_access = true

  depends_on = [google_project_service.required_apis]
}
