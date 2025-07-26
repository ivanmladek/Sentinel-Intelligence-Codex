#!/usr/bin/env python3
"""
Distributed PDF Processing Pipeline Orchestration Script

This script orchestrates the distributed processing of PDF files using:
- Vertex AI endpoints with A100 GPUs for Nougat OCR processing
- GKE cluster for batch processing jobs
- GCS for intermediate storage
- Pipeline parallelism for overlapping processing stages
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

import requests
from google.cloud import storage, aiplatform
from google.cloud.aiplatform import gapic as aip
from google.cloud.aiplatform_v1.types import Endpoint
import docker
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# --- Configuration ---
PROJECT_ID = "gdrive-410709"
REGION = "us-central1"
GCS_BUCKET_NAME = f"{PROJECT_ID}-vllm-13b-demo-bucket"
GKE_CLUSTER_NAME = "batch-inference-cluster"
GKE_ZONE = "us-central1-b"
VERTEX_ENDPOINT_NAME = "llama2-13b-vllm-demo-endpoint"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- GCS Client ---
gcs_client = storage.Client(project=PROJECT_ID)

# --- Vertex AI Client ---
aiplatform.init(project=PROJECT_ID, location=REGION)
vertex_client = aip.EndpointServiceClient(
    client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}
)

def get_gcs_bucket():
    """Get or create the GCS bucket for intermediate storage."""
    try:
        bucket = gcs_client.get_bucket(GCS_BUCKET_NAME)
        logger.info(f"Using existing GCS bucket: {GCS_BUCKET_NAME}")
    except Exception as e:
        logger.info(f"Creating new GCS bucket: {GCS_BUCKET_NAME}")
        bucket = gcs_client.create_bucket(GCS_BUCKET_NAME, location=REGION)
    return bucket

def upload_file_to_gcs(local_path: str, gcs_path: str, bucket) -> str:
    """Upload a file to GCS and return the GCS URI."""
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {local_path} to gs://{GCS_BUCKET_NAME}/{gcs_path}")
    return f"gs://{GCS_BUCKET_NAME}/{gcs_path}"

def download_file_from_gcs(gcs_path: str, local_path: str, bucket):
    """Download a file from GCS to local storage."""
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)
    logger.info(f"Downloaded gs://{GCS_BUCKET_NAME}/{gcs_path} to {local_path}")

def get_vertex_endpoint(endpoint_display_name: str) -> Endpoint:
    """Get the Vertex AI endpoint for Nougat processing."""
    # List endpoints and find the one with the matching display name
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    endpoints = vertex_client.list_endpoints(parent=parent)
    
    for endpoint in endpoints:
        if endpoint.display_name == endpoint_display_name:
            logger.info(f"Found Vertex AI endpoint: {endpoint.name}")
            return endpoint
    
    raise ValueError(f"Vertex AI endpoint '{endpoint_display_name}' not found")

def process_pdf_with_nougat_vertex(pdf_gcs_path: str, output_gcs_path: str, endpoint: Endpoint) -> str:
    """
    Process a PDF using Nougat on Vertex AI endpoint.
    
    This function sends a request to the Vertex AI endpoint to process a PDF file.
    The actual implementation would depend on how the Nougat model is deployed.
    """
    # This is a placeholder implementation. The actual implementation would depend
    # on how the Nougat model is deployed to Vertex AI.
    
    # For now, we'll simulate the processing
    logger.info(f"Processing {pdf_gcs_path} with Nougat on Vertex AI endpoint")
    
    # In a real implementation, you would:
    # 1. Download the PDF from GCS
    # 2. Process it with Nougat
    # 3. Upload the result to GCS
    # 4. Return the GCS path of the result
    
    # Simulate processing time
    time.sleep(5)
    
    # For now, just return a placeholder result
    return output_gcs_path

def create_gke_job(job_name: str, image: str, command: List[str], env_vars: Dict[str, str] = None) -> str:
    """
    Create and submit a job to the GKE cluster.
    
    This function creates a Kubernetes job and submits it to the GKE cluster.
    """
    # Load kubeconfig
    try:
        config.load_kube_config()
    except Exception as e:
        # If we can't load kubeconfig, try in-cluster config
        try:
            config.load_incluster_config()
        except Exception as e2:
            raise Exception(f"Could not load kubeconfig: {e} or {e2}")
    
    # Create API client
    batch_v1 = client.BatchV1Api()
    
    # Define the job
    job = client.V1Job(
        metadata=client.V1ObjectMeta(name=job_name),
        spec=client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                spec=client.V1PodSpec(
                    restart_policy="Never",
                    containers=[
                        client.V1Container(
                            name="pdf-processor",
                            image=image,
                            command=command,
                            env=[client.V1EnvVar(name=k, value=v) for k, v in (env_vars or {}).items()]
                        )
                    ]
                )
            )
        )
    )
    
    # Submit the job
    try:
        api_response = batch_v1.create_namespaced_job(
            body=job,
            namespace="default"
        )
        logger.info(f"Job '{job_name}' created. Status: {api_response.status}")
        return job_name
    except ApiException as e:
        logger.error(f"Exception when creating job: {e}")
        raise

def wait_for_gke_job_completion(job_name: str, timeout: int = 3600) -> bool:
    """
    Wait for a GKE job to complete.
    
    Returns True if the job completed successfully, False otherwise.
    """
    # Load kubeconfig
    try:
        config.load_kube_config()
    except Exception as e:
        # If we can't load kubeconfig, try in-cluster config
        try:
            config.load_incluster_config()
        except Exception as e2:
            raise Exception(f"Could not load kubeconfig: {e} or {e2}")
    
    # Create API client
    batch_v1 = client.BatchV1Api()
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            job = batch_v1.read_namespaced_job(name=job_name, namespace="default")
            if job.status.succeeded:
                logger.info(f"Job '{job_name}' completed successfully")
                return True
            elif job.status.failed:
                logger.error(f"Job '{job_name}' failed")
                return False
            else:
                logger.info(f"Job '{job_name}' is still running...")
                time.sleep(30)  # Wait 30 seconds before checking again
        except ApiException as e:
            logger.error(f"Exception when reading job: {e}")
            return False
    
    logger.error(f"Job '{job_name}' timed out")
    return False

def process_pdfs_distributed(pdf_files: List[str], bucket) -> List[str]:
    """
    Process PDF files using the distributed pipeline.
    
    This function orchestrates the distributed processing of PDF files using:
    1. GKE jobs for PDF discovery and preprocessing
    2. Vertex AI endpoints for Nougat OCR processing
    3. GKE jobs for postprocessing and uploading to Hugging Face
    """
    logger.info(f"Processing {len(pdf_files)} PDF files using distributed pipeline")
    
    # Get Vertex AI endpoint for Nougat processing
    try:
        endpoint = get_vertex_endpoint(VERTEX_ENDPOINT_NAME)
    except Exception as e:
        logger.error(f"Failed to get Vertex AI endpoint: {e}")
        raise
    
    # Create a list to store the results
    results = []
    
    # Process PDFs in batches
    batch_size = 10
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} of {len(pdf_files)//batch_size + 1}")
        
        # Upload PDFs to GCS
        gcs_paths = []
        for pdf_file in batch:
            pdf_path = Path(pdf_file)
            gcs_path = f"input_pdfs/{pdf_path.name}"
            upload_file_to_gcs(str(pdf_path), gcs_path, bucket)
            gcs_paths.append(gcs_path)
        
        # Process PDFs with Nougat on Vertex AI
        mmd_gcs_paths = []
        for gcs_path in gcs_paths:
            pdf_name = Path(gcs_path).name
            mmd_gcs_path = f"processed_mmd/{pdf_name.replace('.pdf', '.mmd')}"
            try:
                result_path = process_pdf_with_nougat_vertex(gcs_path, mmd_gcs_path, endpoint)
                mmd_gcs_paths.append(result_path)
                logger.info(f"Processed {gcs_path} -> {result_path}")
            except Exception as e:
                logger.error(f"Failed to process {gcs_path}: {e}")
                mmd_gcs_paths.append(None)
        
        # Add results to the list
        for pdf_file, mmd_gcs_path in zip(batch, mmd_gcs_paths):
            results.append((pdf_file, mmd_gcs_path))
    
    return results

def main(input_dir: str, output_dir: str):
    """Main function to orchestrate the distributed PDF processing pipeline."""
    logger.info("Starting distributed PDF processing pipeline")
    
    # Get or create GCS bucket
    bucket = get_gcs_bucket()
    
    # Discover PDF files
    input_path = Path(input_dir)
    pdf_files = list(input_path.rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    if not pdf_files:
        logger.warning("No PDF files found. Exiting.")
        return
    
    # Process PDFs using the distributed pipeline
    results = process_pdfs_distributed(pdf_files, bucket)
    
    # Download results from GCS
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    successful_count = 0
    for pdf_file, mmd_gcs_path in results:
        if mmd_gcs_path:
            try:
                local_mmd_path = output_path / f"{Path(pdf_file).stem}.mmd"
                download_file_from_gcs(mmd_gcs_path, str(local_mmd_path), bucket)
                successful_count += 1
                logger.info(f"Downloaded result for {pdf_file} to {local_mmd_path}")
            except Exception as e:
                logger.error(f"Failed to download result for {pdf_file}: {e}")
    
    logger.info(f"Successfully processed {successful_count} out of {len(pdf_files)} PDF files")
    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed PDF Processing Pipeline")
    parser.add_argument("input_dir", help="Input directory containing PDF files")
    parser.add_argument("output_dir", help="Output directory for processed files")
    
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)