import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from google.cloud import storage

# --- Configuration ---
PROJECT_ID = "gdrive-410709"
REGION = "us-central1"
GCS_BUCKET_NAME = f"{PROJECT_ID}-vllm-13b-demo-bucket"

# --- GCS Client ---
gcs_client = storage.Client(project=PROJECT_ID)
bucket = gcs_client.bucket(GCS_BUCKET_NAME)

def upload_file_to_gcs(local_path: str, gcs_path: str):
    """Upload a file to GCS."""
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded {local_path} to gs://{GCS_BUCKET_NAME}/{gcs_path}")

def download_file_from_gcs(gcs_path: str, local_path: str):
    """Download a file from GCS."""
    blob = bucket.blob(gcs_path)
    blob.download_to_filename(local_path)
    print(f"Downloaded gs://{GCS_BUCKET_NAME}/{gcs_path} to {local_path}")

def ocr_with_nougat(file_path: Path, output_dir: Path):
    """Performs OCR on a PDF using the Nougat command-line tool and logs a sample of the output."""
    try:
        # Nougat saves the output file with the same stem as the input, but with a .mmd extension.
        expected_output_path = output_dir / f"{file_path.stem}.mmd"
        
        # Get the full path to the nougat script
        nougat_script_path = "/Users/jj/.pyenv/versions/3.11.4/bin/nougat"
        python_interpreter_path = "/Users/jj/.pyenv/versions/3.11.4/bin/python"
        
        # Run the nougat command
        command = [
            python_interpreter_path,
            nougat_script_path,
            "--no-skipping",
            str(file_path),
            "-o",
            str(output_dir)
        ]
        
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        
        if process.returncode != 0:
            return f"  [Error] Nougat process failed for {file_path.name}: {process.stderr}"
        
        # Verify that the output file was created
        if expected_output_path.exists():
            with expected_output_path.open('r', encoding='utf-8') as f:
                sample_text = f.read(100).replace('\n', ' ')
            return f"  [Success] Nougat created '{expected_output_path.name}'. Sample: {sample_text}..."
        else:
            return f"  [Error] Nougat finished but the output file '{expected_output_path.name}' was not found."
            
    except Exception as e:
        return f"  [Error] An exception occurred during Nougat OCR for {file_path.name}: {e}"

def discover_files(input_dir: Path) -> List[Path]:
    """Recursively discovers all PDF files in the input directory."""
    return list(input_dir.rglob("*.pdf"))

def process_single_pdf_distributed(pdf_file: Path, temp_dir: Path) -> Tuple[Path, str]:
    """
    Process a single PDF file in a distributed manner.
    
    This function:
    1. Uploads the PDF to GCS
    2. Processes it with Nougat (in a distributed worker)
    3. Downloads the result from GCS
    """
    # Upload PDF to GCS
    pdf_gcs_path = f"input_pdfs/{pdf_file.name}"
    upload_file_to_gcs(str(pdf_file), pdf_gcs_path)
    
    # In a real distributed system, we would submit a job to process this PDF
    # For now, we'll process it locally but store intermediate files in GCS
    
    # Download PDF from GCS to temp directory for processing
    temp_pdf_path = temp_dir / pdf_file.name
    download_file_from_gcs(pdf_gcs_path, str(temp_pdf_path))
    
    # Process PDF with Nougat
    result = ocr_with_nougat(temp_pdf_path, temp_dir)
    
    # Upload result to GCS
    mmd_file = temp_dir / f"{pdf_file.stem}.mmd"
    if mmd_file.exists():
        mmd_gcs_path = f"processed_mmd/{mmd_file.name}"
        upload_file_to_gcs(str(mmd_file), mmd_gcs_path)
        return (pdf_file, mmd_gcs_path)
    else:
        return (pdf_file, None)

def main(input_dir: Path, output_dir: Path, temp_dir: Path):
    output_dir.mkdir(exist_ok=True)
    temp_dir.mkdir(exist_ok=True)
    print(f"Output will be saved in: {output_dir.resolve()}")
    print(f"Temp files will be stored in: {temp_dir.resolve()}")
    
    all_files = discover_files(input_dir)
    
    if not all_files:
        print("No PDF files found in the input directory.")
        return
    
    files_to_process = []
    skipped_files = []
    for file_path in all_files:
        expected_output_path = output_dir / f"{file_path.stem}.mmd"
        if not (expected_output_path.exists() and expected_output_path.stat().st_size > 0):
            files_to_process.append(file_path)
        else:
            skipped_files.append(file_path)
    
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files because they already have a non-empty output file.")
    
    if not files_to_process:
        print("All PDF files have already been processed.")
        return
    
    print(f"Processing {len(files_to_process)} out of {len(all_files)} PDF files with Nougat." )
    
    # Process files with GCS intermediate storage
    results = []
    with tqdm(total=len(files_to_process), desc="Overall Progress", unit="file") as pbar:
        for file_path in files_to_process:
            result = process_single_pdf_distributed(file_path, temp_dir)
            results.append(result)
            pbar.update(1)
    
    # Download results from GCS
    successful_count = 0
    for pdf_file, mmd_gcs_path in results:
        if mmd_gcs_path:
            try:
                local_mmd_path = output_dir / f"{pdf_file.stem}.mmd"
                download_file_from_gcs(mmd_gcs_path, str(local_mmd_path))
                successful_count += 1
                pbar.write(f"  [Success] Downloaded result for {pdf_file.name}")
            except Exception as e:
                pbar.write(f"  [Error] Failed to download result for {pdf_file.name}: {e}")
        else:
            pbar.write(f"  [Error] No result for {pdf_file.name}")
    
    print(f"Successfully processed {successful_count} out of {len(files_to_process)} PDF files")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python distributed_extract_text.py <input_directory> [output_directory] [temp_directory]", file=sys.stderr)
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("extracted_output")
    temp_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("temp")
    
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)
    
    main(input_path, output_path, temp_path)