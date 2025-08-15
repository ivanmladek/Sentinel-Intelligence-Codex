# Debugging Process for GKE Job Deployment

## Initial Issues

1. **Docker Image Platform Mismatch**: The initial Docker image was built for ARM64 architecture, but the GKE cluster was running on AMD64 nodes. This caused "no match for platform in manifest" errors when trying to pull the image.

2. **Incorrect Dockerfile Used**: We were initially trying to build and push the `text_classification` Dockerfile instead of the `processing` Dockerfile that was needed for the PDF processing job.

3. **Missing Hugging Face Token Secret**: The job configuration required a Hugging Face token secret, but it wasn't properly configured in the cluster.

4. **Insufficient GPU Resources**: We initially had only one node with one GPU, but our job required four GPUs (one per pod).

5. **Node Pool Configuration**: The node pool needed to be properly configured with GPU resources and preemptible instances.

## Resolution Steps

### 1. Correct Docker Image Building
- Identified that we needed to use `docker/Dockerfile.processing` instead of `docker/Dockerfile.text_classification`
- Rebuilt the image with the correct platform architecture:
  ```
  docker build --platform linux/amd64 -t us-central1-docker.pkg.dev/gdrive-410709/vllm-serving-repo/pdf-processor:latest -f docker/Dockerfile.processing .
  ```
- Pushed the correctly built image to Artifact Registry

### 2. Node Pool Configuration
- Verified that the Terraform configuration was correctly set up with:
  - `nvidia-tesla-t4` GPUs
  - Preemptible instances
  - Proper machine type (`n1-standard-4`)
- Resized the node pool to ensure we had enough nodes:
  ```
  gcloud container clusters resize batch-inference-cluster --node-pool=batch-inference-cluster-gpu-node-pool --num-nodes=4 --zone=us-central1-c
  ```

### 3. Secret Management
- Created the Hugging Face token secret using the existing `kubernetes/huggingface-secret.yaml` file:
  ```
  kubectl apply -f kubernetes/huggingface-secret.yaml
  ```

### 4. Job Configuration
- Updated `gke_job.yaml` to:
  - Use the correct image name (`pdf-processor` instead of `text-classifier`)
  - Request GPU resources (`nvidia.com/gpu: 1`)
  - Include image pull secrets

### 5. Job Deployment
- Deleted the old job and recreated it with the updated configuration:
  ```
  kubectl delete job pdf-processing-job
  kubectl apply -f gke_job.yaml
  ```

## Final Verification

After implementing these changes, the job pods were successfully scheduled and started processing data. The logs showed that the pods were:
- Downloading NLTK data (with some temporary network issues that resolved)
- Accessing the book repository URLs
- Downloading and extracting RAR files

## Key Lessons Learned

1. **Platform Architecture Matters**: Always ensure that Docker images are built for the correct target architecture.

2. **Resource Requirements**: Make sure the cluster has sufficient resources (especially GPUs) for the job requirements.

3. **Secret Management**: Properly configure all required secrets before deploying jobs that depend on them.

4. **Configuration Verification**: Double-check all configuration files to ensure they match the intended deployment.

5. **Incremental Testing**: Test each component separately before deploying the full solution.