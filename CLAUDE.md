# Plan for Distributing PDF Processing Pipeline with Existing Nougat

## Current Architecture Analysis

The current `process_refactor.ipynb` implements a sequential pipeline for processing PDF books:
1. Discovers and downloads RAR files containing PDFs
2. Extracts PDFs from RAR archives
3. Processes each PDF with Nougat OCR (as-is, no modifications)
4. Cleans and filters extracted text
5. Chunks text into appropriate sizes
6. Uploads processed data to Hugging Face

## Target Distributed Architecture

Leveraging the Terraform-deployed infrastructure:
- **Vertex AI Endpoints (2x A100 GPUs)**: For running multiple Nougat instances
- **GKE Cluster**: For orchestrating distributed batch processing jobs
- **Cloud Run Service**: For API/frontend services
- **GCS Bucket**: For intermediate storage of files
- **Artifact Registry**: For container images

## Simplified Transformation Plan

### 1. Preserve Nougat As-Is
- Maintain exact Nougat dependencies without modifications
- numpy==1.26.4
- transformers==4.38.2
- All other original dependencies
- No custom XLA, Triton, or JAX optimizations

### 2. Containerize with Original Nougat
Create Docker containers with:
- Exact Nougat dependencies preserved
- CUDA support for GPU acceleration
- NCCL for inter-container communication
- All necessary system dependencies

### 3. Implement Distributed Processing Strategy

#### For Nougat OCR Processing (Using Existing Nougat):
- Deploy unmodified Nougat to Vertex AI endpoints (2x A100s)
- Run multiple Nougat instances in parallel
- Process multiple PDFs concurrently across GPUs
- Use NCCL for communication between distributed workers

#### For Pipeline Orchestration:
- Implement pipeline parallelism to overlap different processing stages
- Stage 1: Download/extract RAR files (CPU-bound, can be parallelized)
- Stage 2: Distribute PDFs to Nougat processing (GPU-bound, Vertex AI)
- Stage 3: Text cleaning/chunking (CPU-bound, can be parallelized)
- Stage 4: Upload to Hugging Face (I/O-bound)

### 4. Data Management Strategy
- Store raw RAR files in GCS
- Store extracted PDFs in GCS
- Store intermediate MMD files in GCS
- Store final JSONL files in GCS before Hugging Face upload
- Use GCS for efficient data transfer between distributed components

### 5. Orchestration with Basic Parallelism

#### GKE-Based Orchestration:
- Create Kubernetes jobs for each processing stage
- Use pipeline parallelism to run stages concurrently
- Implement job dependencies (extract -> OCR -> clean -> upload)
- Use Vertex AI integration for Nougat processing

#### Where Technologies Are Used:
- **NCCL**: For communication between distributed processing workers
- **Pipeline Parallelism**: For overlapping different processing stages
- **No XLA/Triton/JAX**: Nougat runs as original implementation

### 6. Implementation Steps

1. **Containerize with Preserved Nougat Dependencies**
   - Create Docker image with exact Nougat dependencies
   - Include CUDA support for GPU acceleration
   - Add NCCL for inter-container communication
   - Preserve all original dependency versions

2. **Deploy Unmodified Nougat to Vertex AI**
   - Package Nougat with all dependencies as-is
   - Deploy to Vertex AI endpoints (2x A100s)
   - Configure for batch processing of multiple PDFs

3. **Implement Pipeline Parallelism**
   - Create GKE jobs for each processing stage
   - Implement concurrent execution of independent stages
   - Add proper dependencies between stages
   - Overlap I/O-bound and CPU-bound stages

4. **Develop Orchestration Layer**
   - Simple orchestrator to manage job creation and dependencies
   - Monitor job status and handle failures
   - Manage data flow between stages using GCS
   - Distribute PDFs to available Nougat workers

5. **Implement NCCL Communication**
   - Use NCCL for efficient communication between distributed workers
   - Coordinate work distribution across multiple Nougat instances
   - Synchronize status updates between components

6. **Deploy and Test**
   - Deploy all components to GKE and Vertex AI
   - Test with sample RAR files
   - Verify parallel processing works correctly
   - Optimize job scheduling based on results

This approach focuses on distributing the existing pipeline without modifying Nougat:
- Keep Nougat exactly as it is with original dependencies
- Use pipeline parallelism to overlap different processing stages
- Use NCCL for communication between distributed workers
- Run multiple PDFs through Nougat concurrently
- Add custom speedups later if needed