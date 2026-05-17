# Google Cloud Healthcare API Dataset
resource "google_healthcare_dataset" "oncoflow_dataset" {
  name     = "oncoflow-dataset"
  location = var.region
  time_zone = "UTC"

  depends_on = [google_project_service.required_apis]
}

# Google Cloud Healthcare API DICOM Store
resource "google_healthcare_dicom_store" "dicom_store" {
  name    = "oncoflow-dicom-store"
  dataset = google_healthcare_dataset.oncoflow_dataset.id

  # We don't need notification config or complex setup for the MVP, 
  # just the core DICOM store.
}
