
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
  default     = "us-central1-b"
}

variable "gcs_bucket_name" {
  description = "The name of the GCS bucket for model artifacts."
  type        = string
  default     = "gdrive-410709-vllm-13b-demo-bucket"
}

variable "model_display_name" {
  description = "The display name for the Vertex AI model."
  type        = string
  default     = "llama2-13b-vllm-demo"
}

variable "artifact_registry_repo" {
  description = "The name of the Artifact Registry repository."
  type        = string
  default     = "vllm-serving-repo"
}

variable "cloud_run_service_name" {
  description = "The name of the Cloud Run service for the frontend."
  type        = string
  default     = "langchain-llama2-frontend"
}

variable "gke_cluster_name" {
  description = "The name of the GKE cluster for batch jobs."
  type        = string
  default     = "batch-inference-cluster"
}

variable "machine_type" {
  description = "The machine type for the Vertex AI endpoint."
  type        = string
  default     = "a2-highgpu-2g"
}

variable "accelerator_type" {
  description = "The accelerator type for the Vertex AI endpoint."
  type        = string
  default     = "NVIDIA_TESLA_A100"
}

variable "accelerator_count" {
  description = "The number of accelerators for the Vertex AI endpoint."
  type        = number
  default     = 2
}

variable "vllm_image_uri" {
  description = "The full URI of the vLLM serving container image in Artifact Registry."
  type        = string
  # This must be built and pushed manually before applying the model resource.
  # Example: us-central1-docker.pkg.dev/gdrive-410709/vllm-serving-repo/llama2-13b-vllm-demo:v1
  default     = "us-central1-docker.pkg.dev/gdrive-410709/vllm-serving-repo/llama2-13b-vllm-demo:v1"
}

variable "app_image_uri" {
  description = "The full URI of the frontend app container image in Artifact Registry."
  type        = string
  # This must be built and pushed manually before applying the service resource.
  # Example: us-central1-docker.pkg.dev/gdrive-410709/vllm-serving-repo/langchain-llama2-frontend:latest
  default     = "us-central1-docker.pkg.dev/gdrive-410709/vllm-serving-repo/langchain-llama2-frontend:latest"
}
