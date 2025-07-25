# Main Terraform configuration for the Vertex AI and GKE infrastructure

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.51.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = ">= 4.51.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ------------------------------------------------------------------------------
# API Services
# ------------------------------------------------------------------------------
resource "google_project_service" "apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "artifactregistry.googleapis.com",
    "run.googleapis.com",
    "container.googleapis.com",
    "cloudbuild.googleapis.com"
  ])
  service                    = each.key
  disable_dependent_services = true
}

# ------------------------------------------------------------------------------
# GCS Bucket for Model Artifacts
# ------------------------------------------------------------------------------
resource "google_storage_bucket" "model_bucket" {
  name          = var.gcs_bucket_name
  location      = var.region
  force_destroy = true # Set to false in production
  uniform_bucket_level_access = true
  depends_on = [google_project_service.apis]
}

# ------------------------------------------------------------------------------
# Artifact Registry for Docker Images
# ------------------------------------------------------------------------------
resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = var.artifact_registry_repo
  format        = "DOCKER"
  description   = "Docker repository for vLLM serving containers"
  depends_on = [google_project_service.apis]
}

# ------------------------------------------------------------------------------
# GKE Cluster for Batch Inference
# ------------------------------------------------------------------------------
resource "google_container_cluster" "gke_cluster" {
  name     = var.gke_cluster_name
  location = var.gke_zone
  timeouts {
    create = "60m"
  }

  # Using a simple node pool for batch jobs as in the script
  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false
  depends_on = [google_project_service.apis]
}

resource "google_container_node_pool" "gke_node_pool" {
  name       = "${var.gke_cluster_name}-node-pool"
  location   = var.gke_zone
  cluster    = google_container_cluster.gke_cluster.name
  node_count = 1

  node_config {
    machine_type = "e2-standard-2"
    service_account = "416399941568-compute@developer.gserviceaccount.com"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

# ------------------------------------------------------------------------------
# Cloud Run Frontend Service
# ------------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "frontend_service" {
  name     = var.cloud_run_service_name
  location = var.region
  
  template {
    containers {
      image = var.app_image_uri
      env {
        name = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name = "GCP_REGION"
        value = var.region
      }
    }
  }
  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "frontend_service_invoker" {
  location = google_cloud_run_v2_service.frontend_service.location
  name     = google_cloud_run_v2_service.frontend_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}