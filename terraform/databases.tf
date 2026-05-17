# Google Cloud SQL (PostgreSQL)
resource "google_sql_database_instance" "postgres_instance" {
  name             = "oncoflow-db-instance"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier = "db-f1-micro" # Smallest tier for MVP
    
    # Use public IP for MVP ease of access, 
    # but restrict authorized networks if possible. 
    # In a real setup, we'd use private IP with VPC.
    ip_configuration {
      ipv4_enabled = true
    }
  }

  deletion_protection = false

  depends_on = [google_project_service.required_apis]
}

resource "google_sql_user" "postgres_user" {
  name     = "oncoflow_user"
  instance = google_sql_database_instance.postgres_instance.name
  password = var.db_password
}

resource "google_sql_database" "postgres_db" {
  name     = "oncoflow"
  instance = google_sql_database_instance.postgres_instance.name
}

# Google Cloud Memorystore (Redis)
resource "google_redis_instance" "redis_instance" {
  name           = "oncoflow-redis"
  tier           = "BASIC" # Basic tier is fine for MVP without HA
  memory_size_gb = 1       # Smallest memory size

  region = var.region

  # By default, Memorystore requires a VPC. We will use the default VPC for MVP.
  # authorized_network = "projects/${var.project_id}/global/networks/default"

  depends_on = [google_project_service.required_apis]
}
