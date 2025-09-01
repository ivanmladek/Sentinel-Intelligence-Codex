# Variables for the Terraform configuration

variable "project_id" {
  description = "The GCP project ID to deploy to."
  type        = string
  default     = "gdrive-410709"
}

variable "region" {
  description = "The GCP region for the resources."
  type        = string
  default     = "us-central1"
}

variable "gke_zone" {
  description = "The GCP zone for the GKE cluster."
  type        = string
  default     = "us-central1-c"
}

variable "gcs_bucket_name" {
  description = "The name of the GCS bucket for model artifacts."
  type        = string
  default     = "gdrive-410709-vllm-13b-demo-bucket"
}

variable "artifact_registry_repo" {
  description = "The name of the Artifact Registry repository."
  type        = string
  default     = "vllm-serving-repo"
}

variable "gke_cluster_name" {
  description = "The name of the GKE cluster for batch jobs."
  type        = string
  default     = "batch-inference-cluster"
}

variable "gke_node_pool_machine_type" {
  description = "The machine type for the GKE node pool."
  type        = string
  default     = "n1-standard-4"
}

variable "gke_node_pool_accelerator_type" {
  description = "The accelerator type for the GKE node pool."
  type        = string
  default     = "nvidia-tesla-t4"
}

variable "gke_node_pool_accelerator_count" {
  description = "The number of accelerators for the GKE node pool."
  type        = number
  default     = 4
}