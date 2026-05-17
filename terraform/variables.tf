variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "region" {
  description = "The region to deploy resources in"
  type        = string
  default     = "us-central1"
}

variable "db_password" {
  description = "Password for the Cloud SQL PostgreSQL database"
  type        = string
  sensitive   = true
}
