
# Main Terraform configuration for the Vertex AI and GKE infrastructure

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

  # Using a simple node pool for batch jobs as in the script
  remove_default_node_pool = true
  initial_node_count       = 1
  depends_on = [google_project_service.apis]
}

resource "google_container_node_pool" "gke_node_pool" {
  name       = "${var.gke_cluster_name}-node-pool"
  location   = var.gke_zone
  cluster    = google_container_cluster.gke_cluster.name
  node_count = 1

  node_config {
    machine_type = "e2-standard-2"
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

# ------------------------------------------------------------------------------
# Vertex AI Resources
# ------------------------------------------------------------------------------
resource "google_vertex_ai_model" "vertex_model" {
  display_name = var.model_display_name
  region       = var.region
  description  = "LLaMA2 13B model served with vLLM"

  # This model resource is created with a serving container, but no artifact
  # URI, as the model files are baked into the container itself.
  container_spec {
    image_uri = var.vllm_image_uri
    # The script doesn't specify command/args, assuming they are in the Dockerfile
  }
  depends_on = [google_project_service.apis]
}

resource "google_vertex_ai_endpoint" "vertex_endpoint" {
  name         = "${var.model_display_name}-endpoint"
  display_name = "${var.model_display_name}-endpoint"
  region       = var.region
  project      = var.project_id
  depends_on = [google_project_service.apis]
}

# Deploys the model to the endpoint
resource "google_vertex_ai_model_deployment" "model_deployment" {
  endpoint = google_vertex_ai_endpoint.vertex_endpoint.id
  model    = google_vertex_ai_model.vertex_model.id
  
  dedicated_resources {
    machine_spec {
      machine_type     = var.machine_type
      accelerator_type = var.accelerator_type
      accelerator_count = var.accelerator_count
    }
    min_replica_count = 1
    max_replica_count = 1 # Can be increased for auto-scaling
  }

  # This depends on the model being created first
  depends_on = [google_vertex_ai_model.vertex_model]
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
        name  = "VERTEX_ENDPOINT_ID"
        value = split("/", google_vertex_ai_endpoint.vertex_endpoint.id)[5]
      }
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

  # Allow unauthenticated access as in the script
  iam_bindings {
    role = "roles/run.invoker"
    members = [
      "allUsers",
    ]
  }
  depends_on = [google_project_service.apis, google_vertex_ai_endpoint.vertex_endpoint]
}
