terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# Enable required GCP APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "run.googleapis.com",
    "aiplatform.googleapis.com",
    "healthcare.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
  ])

  service            = each.key
  disable_on_destroy = false
}
