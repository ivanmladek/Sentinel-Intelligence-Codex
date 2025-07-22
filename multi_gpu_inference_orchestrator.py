import os
import subprocess
import getpass
from google.cloud import aiplatform

# --- CONFIGURATION ---
# PLEASE REPLACE WITH YOUR VALUES
PROJECT_ID = "gdrive-410709"
REGION = "us-central1"
GCS_BUCKET_NAME = f"{PROJECT_ID}-vllm-13b-demo-bucket"
MODEL_DISPLAY_NAME = "llama2-13b-vllm-demo"

# --- Docker/Cloud Run/GKE Config ---
ARTIFACT_REGISTRY_REPO = "vllm-serving-repo"
CLOUD_RUN_SERVICE_NAME = "langchain-llama2-frontend"
GKE_CLUSTER_NAME = "batch-inference-cluster"
GKE_ZONE = "us-central1-c"

# --- Vertex AI Endpoint Machine Type ---
# Using a2-highgpu-2g which comes with 2 A100 40GB GPUs.
# Ensure you have quota for NVIDIA_TESLA_A100-40GB GPUs.
MACHINE_TYPE = "a2-highgpu-2g"
ACCELERATOR_TYPE = "NVIDIA_TESLA_A100-40GB"
ACCELERATOR_COUNT = 2

def run_cmd(cmd, description, cwd="."):
    """Executes a shell command in a given directory and prints its description."""
    print(f"--- {description} (in ./{cwd}) ---")
    try:
        subprocess.run(cmd, shell=True, check=True, text=True, cwd=cwd)
        print(f"--- SUCCESS: {description} ---\n")
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR in '{description}': {e} ---")
        raise

def main():
    """Main orchestration function."""
    # --- 0. Initial Setup ---
    hf_token = getpass.getpass("Please enter your Hugging Face Read Token: ")
    if not hf_token:
        raise ValueError("Hugging Face token is required.")

    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{GCS_BUCKET_NAME}")

    run_cmd(f"gcloud config set project {PROJECT_ID}", "Set gcloud project")
    run_cmd(f"gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com run.googleapis.com container.googleapis.com", "Enable required GCP APIs")
    run_cmd(f"gsutil mb -p {PROJECT_ID} -l {REGION} gs://{GCS_BUCKET_NAME}", f"Create GCS Bucket (if it doesn't exist)")

    # --- 1. Build and Push Custom vLLM Serving Container ---
    print("--- Building and Pushing vLLM Serving Container ---")
    run_cmd(f"gcloud artifacts repositories create {ARTIFACT_REGISTRY_REPO} --repository-format=docker --location={REGION} --description='vLLM Serving Repo'", "Create Artifact Registry Repo")
    run_cmd(f"gcloud auth configure-docker {REGION}-docker.pkg.dev", "Configure Docker Auth")

    vllm_image_uri_v1 = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/{MODEL_DISPLAY_NAME}:v1"
    build_cmd = f"docker build -f Dockerfile -t {vllm_image_uri_v1} --build-arg HF_TOKEN={hf_token} ."
    run_cmd(build_cmd, "Build vLLM container (this may take several minutes)", cwd="serving")
    run_cmd(f"docker push {vllm_image_uri_v1}", "Push vLLM container to Artifact Registry")

    # --- 2. Register and Deploy Model from Container ---
    print("--- Registering and Deploying Model in Vertex AI ---")
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        serving_container_image_uri=vllm_image_uri_v1,
        serving_container_health_route="/health",
        serving_container_predict_routes="/v1/completions",
        serving_container_ports=[8080],
    )
    print(f"--- SUCCESS: Model registered: {model.resource_name} ---")

    endpoint = model.deploy(
        deployed_model_display_name=MODEL_DISPLAY_NAME,
        machine_type=MACHINE_TYPE,
        accelerator_type=ACCELERATOR_TYPE,
        accelerator_count=ACCELERATOR_COUNT,
        sync=True
    )
    print(f"--- SUCCESS: Model deployed to Endpoint: {endpoint.resource_name} ---\n")

    # --- 3. Setup Model Monitoring ---
    print("--- Setting up Vertex AI Model Monitoring ---")
    monitor_job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name=f"monitor-{MODEL_DISPLAY_NAME}",
        project=PROJECT_ID,
        location=REGION,
        endpoint=endpoint,
        logging_sampling_strategy=aiplatform.sampling.RandomSampleConfig(sample_rate=1.0),
        schedule_config=aiplatform.schedule.CronScheduleConfig(cron="0 */1 * * *"),
        alert_config=aiplatform.alert.EmailAlertConfig(user_emails=["your-email@example.com"], enable_logging=True),
    )
    print(f"--- SUCCESS: Monitoring job created: {monitor_job.resource_name} ---\n")

    # --- 4. Deploy LangChain App to Cloud Run ---
    print("--- Deploying LangChain Frontend to Cloud Run ---")
    app_image_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/{CLOUD_RUN_SERVICE_NAME}:latest"
    run_cmd(f"docker build -f Dockerfile -t {app_image_uri} .", "Build frontend app container", cwd="app")
    run_cmd(f"docker push {app_image_uri}", "Push frontend app container")

    endpoint_id_for_run = endpoint.resource_name.split('/')[-1]
    run_cmd(
        f"gcloud run deploy {CLOUD_RUN_SERVICE_NAME} "
        f"--image={app_image_uri} "
        f"--platform=managed "
        f"--region={REGION} "
        f"--allow-unauthenticated "
        f"--set-env-vars=VERTEX_ENDPOINT_ID={endpoint_id_for_run},GCP_PROJECT_ID={PROJECT_ID},GCP_REGION={REGION}",
        "Deploy container to Cloud Run"
    )
    cloud_run_url = subprocess.check_output(
        f"gcloud run services describe {CLOUD_RUN_SERVICE_NAME} --platform=managed --region={REGION} --format='value(status.url)'",
        shell=True, text=True
    ).strip()
    print(f"--- SUCCESS: Cloud Run service deployed. URL: {cloud_run_url} ---\n")
    print(f"To test, run: curl -X POST -H \"Content-Type: application/json\" -d '{{\"prompt\": \"What is the capital of France?\"}}' {cloud_run_url}/predict")

    # --- 5. Demonstrate GKE for Batch Jobs ---
    print("--- Demonstrating GKE for a simulated batch job ---")
    run_cmd(f"gcloud container clusters create {GKE_CLUSTER_NAME} --zone {GKE_ZONE} --num-nodes=1 --machine-type=e2-standard-2", "Create GKE Cluster")
    run_cmd(f"gcloud container clusters get-credentials {GKE_CLUSTER_NAME} --zone {GKE_ZONE}", "Get GKE credentials")
    run_cmd("kubectl apply -f gke_job.yaml", "Apply Kubernetes Job to GKE cluster")
    print("--- SUCCESS: GKE Job started. Use 'kubectl logs -l job-name=batch-processing-job' to monitor. ---\n")

    # --- 6. Demonstrate Model Versioning and Rollback ---
    print("--- Demonstrating Model Versioning and Rollback ---")
    vllm_image_uri_v2 = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/{MODEL_DISPLAY_NAME}:v2"
    run_cmd(f"docker tag {vllm_image_uri_v1} {vllm_image_uri_v2}", "Tag image as v2")
    run_cmd(f"docker push {vllm_image_uri_v2}", "Push v2 image")

    model_v2 = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        serving_container_image_uri=vllm_image_uri_v2,
        parent_model=model.resource_name,
        is_default_version=False,
        serving_container_health_route="/health",
        serving_container_predict_routes="/v1/completions",
        serving_container_ports=[8080],
    )
    print(f"--- Registered Version 2 of the model: {model_v2.version_id} ---")

    print("--- Updating endpoint to serve Version 2 ---")
    endpoint.deploy(model=model_v2, traffic_split={"0": 100}, sync=True)
    print("--- SUCCESS: Endpoint updated to Version 2. ---")

    print("--- Rolling back to Version 1 ---")
    endpoint.deploy(model=model, traffic_split={"0": 100}, sync=True)
    print("--- SUCCESS: Rollback to Version 1 complete. ---")

    print("\n\n--- ORCHESTRATION COMPLETE ---")
    print("--- CLEANUP INSTRUCTIONS ---")
    print("To avoid ongoing charges, please manually delete the following resources from the GCP Console:")
    print(f"1. Vertex AI Endpoint: {endpoint.display_name}")
    print(f"2. Vertex AI Model: {MODEL_DISPLAY_NAME}")
    print(f"3. Cloud Run Service: {CLOUD_RUN_SERVICE_NAME}")
    print(f"4. GKE Cluster: {GKE_CLUSTER_NAME}")
    print(f"5. Artifact Registry Repo: {ARTIFACT_REGISTRY_REPO}")
    print(f"6. GCS Bucket: {GCS_BUCKET_NAME}")

if __name__ == "__main__":
    main()