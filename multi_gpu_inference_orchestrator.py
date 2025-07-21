

import os
import subprocess
import textwrap
import getpass
from google.cloud import aiplatform

# --- CONFIGURATION ---
# PLEASE REPLACE WITH YOUR VALUES
PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1"
GCS_BUCKET_NAME = f"{PROJECT_ID}-vllm-demo-bucket"
MODEL_DISPLAY_NAME = "llama2-7b-vllm-demo"

# --- Hugging Face Model ---
# We will use Llama-2 7B, which benefits greatly from a 2-GPU setup.
HF_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

# --- Docker/Cloud Run/GKE Config ---
ARTIFACT_REGISTRY_REPO = "vllm-serving-repo"
CLOUD_RUN_SERVICE_NAME = "langchain-llama2-frontend"
GKE_CLUSTER_NAME = "batch-inference-cluster"
GKE_ZONE = "us-central1-c"

# --- Vertex AI Endpoint Machine Type ---
# Using g2-standard-8 which comes with 2 L4 GPUs. This is a cost-effective choice for inference.
# For higher performance, you could use 'a2-highgpu-2g' (2 A100 40GB).
MACHINE_TYPE = "g2-standard-8"
ACCELERATOR_TYPE = "NVIDIA_L4"
ACCELERATOR_COUNT = 2


def run_cmd(cmd, description, env=None):
    """Executes a shell command and prints its description."""
    print(f"--- {description} ---")
    try:
        # Combine the current environment with any additional variables
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        subprocess.run(cmd, shell=True, check=True, text=True, env=full_env)
        print(f"--- SUCCESS: {description} ---\n")
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR in '{description}': {e} ---")
        raise

def create_vllm_dockerfile():
    """Creates the Dockerfile for the vLLM custom serving container."""
    dockerfile_content = textwrap.dedent("""
    # Use an official NVIDIA CUDA base image
    FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

    # Set environment variables to prevent interactive prompts during installation
    ENV DEBIAN_FRONTEND=noninteractive
    ENV TZ=Etc/UTC

    # Install Python, pip, and git
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip git && \
        rm -rf /var/lib/apt/lists/*

    # Install vLLM and other dependencies
    # Note: vLLM is installed from source to ensure compatibility with the CUDA version.
    # For pre-built wheels, see vLLM documentation.
    RUN pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    RUN pip3 install vllm==0.2.6 huggingface_hub

    # Argument to receive the Hugging Face token
    ARG HF_TOKEN
    ENV HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

    # Health check and prediction ports for Vertex AI
    ENV AIP_HEALTH_ROUTE=/health
    ENV AIP_PREDICT_ROUTE=/v1/completions
    ENV AIP_HTTP_PORT=8080

    # Start the vLLM OpenAI-compatible server
    # It will download the model on first startup.
    # TENSOR_PARALLEL_SIZE is the key to splitting the model across GPUs.
    CMD [ "python3", "-m", "vllm.entrypoints.openai.api_server", \
          "--host", "0.0.0.0", \
          "--port", "8080", \
          "--model", "meta-llama/Llama-2-7b-chat-hf", \
          "--tensor-parallel-size", "2" ]
    """)
    with open("Dockerfile.vllm", "w") as f:
        f.write(dockerfile_content)
    print("--- Created local Dockerfile.vllm ---\n")

def create_langchain_app_and_dockerfile():
    """Creates the Flask app, requirements, and Dockerfile for the Cloud Run frontend."""
    app_content = textwrap.dedent(f"""
    import os
    import json
    import requests
    from flask import Flask, request, jsonify
    from google.cloud import secretmanager

    app = Flask(__name__)

    # --- CONFIGURATION ---
    PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
    REGION = os.environ.get("GCP_REGION")
    VERTEX_ENDPOINT_ID = os.environ.get("VERTEX_ENDPOINT_ID")
    # The full URL for the Vertex AI Endpoint prediction
    PREDICT_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{VERTEX_ENDPOINT_ID}:predict"

    def get_gcp_token():
        # Use requests to get the access token from the metadata server
        # This is the standard way to get credentials on GCP compute services
        metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
        response = requests.get(metadata_url, headers={{"Metadata-Flavor": "Google"}})
        response.raise_for_status()
        return response.json()["access_token"]

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({{"error": "Invalid input, 'prompt' field is required."}}), 400

        prompt = data['prompt']
        access_token = get_gcp_token()

        # The vLLM server expects a payload in OpenAI's format
        payload = {{
            "instances": [
                {{
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": 0.7,
                }}
            ]
        }}

        headers = {{
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }}

        try:
            response = requests.post(PREDICT_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an exception for bad status codes
            # The actual completion is nested inside the response
            completion = response.json()["predictions"][0]
            return jsonify({{"response": completion}})
        except Exception as e:
            return jsonify({{"error": f"Failed to call Vertex AI Endpoint: {str(e)}" , "details": response.text}}), 500

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
    """)
    with open("app.py", "w") as f:
        f.write(app_content)

    requirements_content = textwrap.dedent("""
    Flask==2.2.2
    gunicorn==20.1.0
    requests==2.31.0
    google-cloud-secret-manager
    """)
    with open("requirements.txt", "w") as f:
        f.write(requirements_content)

    dockerfile_content = textwrap.dedent("""
    FROM python:3.9-slim
    WORKDIR /app
    COPY requirements.txt requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt
    COPY app.py .
    CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "app:app"]
    """)
    with open("Dockerfile.app", "w") as f:
        f.write(dockerfile_content)
    print("--- Created local app.py, requirements.txt, and Dockerfile.app ---\n")


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

    create_vllm_dockerfile()
    create_langchain_app_and_dockerfile()

    # --- 1. Build and Push Custom vLLM Serving Container ---
    print("--- Building and Pushing vLLM Serving Container ---")
    run_cmd(f"gcloud artifacts repositories create {ARTIFACT_REGISTRY_REPO} --repository-format=docker --location={REGION} --description='vLLM Serving Repo'", "Create Artifact Registry Repo")
    run_cmd(f"gcloud auth configure-docker {REGION}-docker.pkg.dev", "Configure Docker Auth")

    vllm_image_uri_v1 = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/{MODEL_DISPLAY_NAME}:v1"
    build_cmd = f"docker build -f Dockerfile.vllm -t {vllm_image_uri_v1} --build-arg HF_TOKEN={hf_token} ."
    run_cmd(build_cmd, "Build vLLM container (this may take several minutes)")
    run_cmd(f"docker push {vllm_image_uri_v1}", "Push vLLM container to Artifact Registry")

    # --- 2. Register and Deploy Model from Container ---
    print("--- Registering and Deploying Model in Vertex AI ---")
    # Register the container as a model in the Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        serving_container_image_uri=vllm_image_uri_v1,
        serving_container_health_route="/health",
        serving_container_predict_routes="/v1/completions",
        serving_container_ports=[8080],
    )
    print(f"--- SUCCESS: Model registered: {model.resource_name} ---")

    # Deploy the model to an endpoint with 2 L4 GPUs
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
    # Model monitoring for custom containers requires manually logging predictions
    # Here, we just create the monitoring job.
    monitor_job = aiplatform.ModelDeploymentMonitoringJob.create(
        display_name=f"monitor-{MODEL_DISPLAY_NAME}",
        project=PROJECT_ID,
        location=REGION,
        endpoint=endpoint,
        logging_sampling_strategy=aiplatform.sampling.RandomSampleConfig(sample_rate=1.0),
        schedule_config=aiplatform.schedule.CronScheduleConfig(cron="0 */1 * * *"), # Run every hour
        alert_config=aiplatform.alert.EmailAlertConfig(user_emails=["your-email@example.com"], enable_logging=True),
    )
    print(f"--- SUCCESS: Monitoring job created: {monitor_job.resource_name} ---\n")

    # --- 4. Deploy LangChain App to Cloud Run ---
    print("--- Deploying LangChain Frontend to Cloud Run ---")
    app_image_uri = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY_REPO}/{CLOUD_RUN_SERVICE_NAME}:latest"
    run_cmd(f"docker build -f Dockerfile.app -t {app_image_uri} .", "Build frontend app container")
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
    gke_job_yaml = textwrap.dedent("""
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: batch-processing-job
    spec:
      template:
        spec:
          containers:
          - name: batch-worker
            image: gcr.io/google-containers/busybox
            command: ["sh",  "-c", "echo 'Simulating a batch inference job...' && sleep 30 && echo 'Batch job complete.'"]
          restartPolicy: Never
      backoffLimit: 4
    """)
    with open("gke_job.yaml", "w") as f:
        f.write(gke_job_yaml)
    run_cmd(f"gcloud container clusters get-credentials {GKE_CLUSTER_NAME} --zone {GKE_ZONE}", "Get GKE credentials")
    run_cmd("kubectl apply -f gke_job.yaml", "Apply Kubernetes Job to GKE cluster")
    print("--- SUCCESS: GKE Job started. Use 'kubectl logs -l job-name=batch-processing-job' to monitor. ---\n")

    # --- 6. Demonstrate Model Versioning and Rollback ---
    print("--- Demonstrating Model Versioning and Rollback ---")
    # For a new version, we would typically push a new container image with an updated model or code.
    # Here, we'll just re-tag the existing image to simulate this.
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
