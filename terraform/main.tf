# Main Terraform configuration for the GKE infrastructure

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.51.0"
    }
  }
}

provider "google" {
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
    "container.googleapis.com",
    "cloudbuild.googleapis.com"
  ])
  service                    = each.key
  disable_dependent_services = true
}

# ------------------------------------------------------------------------------
# GCS Bucket for Data
# ------------------------------------------------------------------------------
resource "google_storage_bucket" "data_bucket" {
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
  description   = "Docker repository for processing containers"
  depends_on = [google_project_service.apis]
}

# ------------------------------------------------------------------------------
# GKE Cluster for Parallel Processing
# ------------------------------------------------------------------------------
resource "google_container_cluster" "gke_cluster" {
  name     = var.gke_cluster_name
  location = var.gke_zone
  timeouts {
    create = "60m"
  }

  remove_default_node_pool = true
  initial_node_count       = 1
  deletion_protection      = false
  depends_on = [google_project_service.apis]
}

resource "google_container_node_pool" "gpu_node_pool" {
  name       = "${var.gke_cluster_name}-gpu-node-pool"
  location   = var.gke_zone
  cluster    = google_container_cluster.gke_cluster.name
  node_count = 1

  node_config {
    machine_type = var.gke_node_pool_machine_type
    service_account = "416399941568-compute@developer.gserviceaccount.com"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    preemptible = true

    guest_accelerator {
      type = var.gke_node_pool_accelerator_type
      count = var.gke_node_pool_accelerator_count
    }
  }
}
