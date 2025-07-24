
# Outputs for the Terraform configuration

output "gcs_bucket" {
  description = "The name of the GCS bucket created."
  value       = google_storage_bucket.model_bucket.name
}

output "artifact_registry_repository" {
  description = "The name of the Artifact Registry repository created."
  value       = google_artifact_registry_repository.docker_repo.name
}

output "gke_cluster" {
  description = "The name of the GKE cluster created."
  value       = google_container_cluster.gke_cluster.name
}

output "vertex_ai_model" {
  description = "The resource name of the Vertex AI model created."
  value       = google_vertex_ai_model.vertex_model.name
}

output "vertex_ai_endpoint" {
  description = "The resource name of the Vertex AI endpoint created."
  value       = google_vertex_ai_endpoint.vertex_endpoint.name
}

output "cloud_run_service_url" {
  description = "The URL of the deployed Cloud Run frontend service."
  value       = google_cloud_run_v2_service.frontend_service.uri
}
