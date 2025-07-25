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