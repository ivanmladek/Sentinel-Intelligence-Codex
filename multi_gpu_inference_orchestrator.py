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
ACCELERATOR_TYPE = "nvidia-tesla-a100"
ACCELERATOR_COUNT = 2

def run_cmd(cmd, description, cwd="."):
    """Executes a shell command in a given directory and prints its description."""
    print(f"--- {description} (in ./{cwd}) ---")
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, cwd=cwd, capture_output=True)
        print(f"--- SUCCESS: {description} ---\n")
    except subprocess.CalledProcessError as e:
        if "Bucket names must be globally unique" in e.stderr and "gsutil mb" in cmd:
            print(f"--- INFO: GCS Bucket already exists. Skipping creation. ---\n")
        elif "ALREADY_EXISTS" in e.stderr and "gcloud artifacts repositories create" in cmd:
            print(f"--- INFO: Artifact Registry repository already exists. Skipping creation. ---\n")
        else:
            print(f"--- ERROR in '{description}': {e} ---")
            print(f"Stderr: {e.stderr}")
            raise

def main():
    """Main orchestration function."""
    # --- 0. Initial Setup ---
    hf_token = os.getenv("HUGGING_FACE")
    if not hf_token:
        raise ValueError("Hugging Face token is required.")

    import google.auth
    credentials, project = google.auth.default()
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{GCS_BUCKET_NAME}", credentials=credentials)

    run_cmd(f"gcloud config set project {PROJECT_ID}", "Set gcloud project")
    run_cmd(f"gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com run.googleapis.com container.googleapis.com", "Enable required GCP APIs")
    try:
        run_cmd(f"gsutil mb -p {PROJECT_ID} -l {REGION} gs://{GCS_BUCKET_NAME}", f"Create GCS Bucket (if it doesn't exist)")
    except subprocess.CalledProcessError as e:
        if "BucketAlreadyExists" in str(e):
            print(f"--- INFO: GCS Bucket gs://{GCS_BUCKET_NAME} already exists. Skipping creation. ---")
        else:
            raise

    # --- 1. Build and Push Custom vLLM Serving Container ---
    print("--- Building and Pushing vLLM Serving Container ---")
    run_cmd(f"gcloud artifacts repositories create {ARTIFACT_REGISTRY_REPO} --repository-format=docker --location={REGION} --description='vLLM Serving Repo'", "Create Artifact Registry Repo")
    run_cmd(f"gcloud auth configure-docker {REGION}-docker.pkg.dev", "Configure Docker Auth")
    run_cmd(f"gcloud auth print-access-token | docker login --username=oauth2accesstoken --password-stdin https://us-docker.pkg.dev", "Docker Login to Google Artifact Registry")

    vllm_image_uri_v1 = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/{MODEL_DISPLAY_NAME}:v1"
    build_cmd = f"docker build --platform linux/amd64 -f Dockerfile -t {vllm_image_uri_v1} ."
    run_cmd(build_cmd, "Build vLLM container (this may take several minutes)", cwd="serving")
    run_cmd(f"docker push {vllm_image_uri_v1}", "Push vLLM container to Artifact Registry")

    # --- 2. Register and Deploy Model from Container ---
    print("--- Registering and Deploying Model in Vertex AI ---")
    run_cmd(f"gcloud ai models upload --display-name={MODEL_DISPLAY_NAME} --region={REGION} --container-image-uri={vllm_image_uri_v1} --artifact-uri=gs://{GCS_BUCKET_NAME}/empty_model_artifacts", "Upload Model to Vertex AI")
    model_resource_name = subprocess.check_output(f"gcloud ai models list --filter='display_name={MODEL_DISPLAY_NAME}' --format='value(name)' --region={REGION} --limit=1", shell=True, text=True).strip()
    print(f"--- SUCCESS: Model registered: {model_resource_name} ---")

    run_cmd(f"gcloud ai endpoints create --display-name={MODEL_DISPLAY_NAME}-endpoint --region={REGION}", "Create Vertex AI Endpoint")
    endpoint_resource_name = subprocess.check_output(f"gcloud ai endpoints list --filter='display_name={MODEL_DISPLAY_NAME}-endpoint' --format='value(name)' --region={REGION} --limit=1", shell=True, text=True).strip()
    run_cmd(f"gcloud ai endpoints deploy-model {endpoint_resource_name} --region={REGION} --model={model_resource_name} --display-name={MODEL_DISPLAY_NAME} --machine-type={MACHINE_TYPE} --accelerator=type={ACCELERATOR_TYPE},count={ACCELERATOR_COUNT} --traffic-split=0=100", "Deploy Model to Endpoint")
    
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
        alert_config=aiplatform.alert.EmailAlertConfig(user_emails=["monitoring@gdrive-410709.iam.gserviceaccount.com"], enable_logging=True),
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

    
    run_cmd(f"gcloud ai models upload --display-name={MODEL_DISPLAY_NAME} --region={REGION} --container-image-uri={vllm_image_uri_v2} --artifact-uri=gs://{GCS_BUCKET_NAME}/empty_model_artifacts --parent-model={model_resource_name}", "Upload Model Version 2 to Vertex AI")
    model_v2_resource_name = subprocess.check_output(f"gcloud ai models list --filter='display_name={MODEL_DISPLAY_NAME} AND parent_model={model.resource_name}' --format='value(name)' --region={REGION} --limit=1", shell=True, text=True).strip()
    print(f"--- Registered Version 2 of the model: {model_v2_resource_name} ---")

    print("--- Updating endpoint to serve Version 2 ---")
    run_cmd(f"gcloud ai endpoints deploy-model {endpoint_resource_name} --region={REGION} --model={model_v2_resource_name} --display-name={MODEL_DISPLAY_NAME} --traffic-split=0=100", "Update endpoint to serve Version 2")
    print("--- SUCCESS: Endpoint updated to Version 2. ---")

    print("--- Rolling back to Version 1 ---")
    run_cmd(f"gcloud ai endpoints deploy-model {endpoint_resource_name} --region={REGION} --model={model_resource_name} --display-name={MODEL_DISPLAY_NAME} --traffic-split=0=100", "Rollback to Version 1")
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