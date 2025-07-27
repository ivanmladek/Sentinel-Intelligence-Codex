import os
import re
import json
import logging
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from huggingface_hub import HfApi
from langdetect import detect, LangDetectException
from nltk.corpus import words, brown
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from tqdm import tqdm
from google.cloud import storage

# --- Configuration ---
BASE_URL = os.environ.get("BASE_URL", "https://the-eye.eu/public/Books/Bibliotheca%20Alexandrina/")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
HUGGING_FACE_REPO = os.environ.get("HUGGING_FACE_REPO", "Disperser5601/Sentinel-Intelligence-Codex")
GARBAGE_THRESHOLD = 0.5
LENWORD = 50

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NLTK Data ---
nltk.download('punkt')
nltk.download('words')
nltk.download('punkt_tab')

# --- Helper Functions (from notebook) ---

def get_file_list(url, depth=0, max_depth=3):
    """Recursively get a list of files from a URL and its subdirectories up to a max depth, avoiding backlinks."""
    if depth > max_depth:
        logger.debug(f"Max depth ({max_depth}) reached at URL: {url}. Stopping recursion.")
        return []

    rar_files = []
    retry_strategy = requests.packages.urllib3.util.retry.Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("http://", adapter)
    http.mount("https://", adapter)

    logger.info(f"Accessing URL: {url} (Depth: {depth})")
    try:
        response = http.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                full_url = requests.compat.urljoin(url, href)
                if full_url.startswith(url) and len(full_url) > len(url) and full_url.endswith('/'):
                     logger.debug(f"Found subdirectory: {full_url}. Recursing.")
                     rar_files.extend(get_file_list(full_url, depth + 1, max_depth))
                elif full_url.endswith('.rar'):
                    logger.debug(f"Found RAR file: {full_url}")
                    rar_files.append(full_url)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error accessing URL {url}: {e}")
    logger.debug(f"Finished processing URL: {url}. Found {len(rar_files)} RAR files in this branch.")
    return rar_files

def download_file(url, output_path):
    """Download a file from a URL."""
    if os.path.exists(output_path):
        logger.info(f"{output_path} already exists. Skipping download.")
        return True
    logger.info(f"Attempting to download {url} to {output_path}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Successfully downloaded {url} to {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file from {url}: {e}")
        return False

def extract_rar(file_path, output_path):
    """Extract a RAR file."""
    if not os.path.exists(file_path):
        logger.error(f"RAR file not found for extraction: {file_path}")
        return False
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.debug(f"Created output directory for extraction: {output_path}")
    logger.info(f"Attempting to extract {file_path} to {output_path}")
    try:
        result = subprocess.run(['unrar', 'x', '-o+', file_path, output_path], check=True, capture_output=True, text=True)
        logger.info(f"Successfully extracted {file_path} to {output_path}")
        if result.stdout:
            logger.debug(f"Unrar stdout for {file_path}:\n{result.stdout}")
        if result.stderr:
             logger.debug(f"Unrar stderr for {file_path}:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting {file_path}: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Unrar command not found. Please ensure 'unrar' is installed.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during extraction of {file_path}: {e}")
        return False

def sanitize_filename(filename):
    """Sanitize a filename."""
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    logger.debug(f"Sanitized filename '{filename}' to '{sanitized}'")
    return sanitized

def process_pdf(pdf_path, output_dir):
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found for processing: {pdf_path}")
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.debug(f"Created output directory for Nougat: {output_dir}")

    pdf_filename = os.path.basename(pdf_path)
    expected_mmd_filename = f"{os.path.splitext(pdf_filename)[0]}.mmd"
    mmd_path = os.path.join(output_dir, expected_mmd_filename)

    if os.path.exists(mmd_path):
        logger.info(f"{mmd_path} already exists. Skipping Nougat processing for {pdf_path}.")
        return mmd_path

    logger.info(f"Attempting to process PDF: {pdf_path} with Nougat. Output to {output_dir}")

    try:
        process = subprocess.Popen(
            ['nougat', pdf_path, '-o', output_dir, '--batchsize', '4', '--no-skipping'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )

        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        process.stdout.close()
        return_code = process.wait()

        if return_code != 0:
            logger.error(f"Nougat process failed with exit code {return_code}")
            return None

        logger.info(f"Nougat command finished with exit code {return_code}. Checking for output file.")

        if os.path.exists(mmd_path):
            logger.info(f"Successfully processed {pdf_path} with Nougat. MMD file created at {mmd_path}.")
            return mmd_path
        else:
            logger.error(f"Nougat command finished but expected output {mmd_path} not found.")
            return None

    except Exception as e:
        logger.error(f"An error occurred during Nougat processing of {pdf_path}: {e}")
        return None

def check_gcs_file_exists(bucket_name, blob_name):
    """Check if a file exists in Google Cloud Storage."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logger.error(f"Error checking GCS file existence for {blob_name}: {e}")
        return False

def upload_to_huggingface(file_path, repo_id, repo_type="dataset"):
    """Upload a file to a Hugging Face repository."""
    if not os.path.exists(file_path):
        logger.error(f"File not found for upload to Hugging Face: {file_path}")
        return False

    # Get Hugging Face token from environment variable
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        logger.error("HUGGING_FACE_TOKEN environment variable not set. Cannot upload to Hugging Face.")
        return False

    logger.info(f"Attempting to upload {file_path} to Hugging Face repo '{repo_id}' (type: {repo_type}).")
    try:
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            repo_type=repo_type,
        )
        logger.info(f"Successfully uploaded {file_path} to {repo_id}")
        return True
    except Exception as e:
        logger.error(f"Error uploading {file_path} to Hugging Face repo '{repo_id}': {e}")
        return False

def clean_text(text):
    logger.debug(f"Cleaning text (first 100 chars): {text[:100]}...")
    initial_len = len(text)
    text = re.sub(r'^\s*#+\s+.*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\**|__)(.*?)\1', r'\2', text)
    text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*\)', '', text)
    text = re.sub(r'Table\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Figure\s+\d+\.\d+\s*[\.,;:]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\begin\{.*?\}\s*.*?\\end\{.*?\}', '', text, flags=re.DOTALL)
    text = re.sub(r'\\[a-zA-Z]+\{.*?\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+\s', ' ', text)
    text = re.sub(r'\$.*?\$', '', text)
    text = re.sub(r'\$\$.*?\$\$', '', text)
    text = re.sub(r'\\\\[.*?\\\\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*\\\)', '', text, flags=re.DOTALL)
    text = text.replace('\\\\', ' ')
    text = text.replace('\\%', '%')
    text = text.replace('\\&', '&')
    text = text.replace('\\$', '$')
    text = text.replace('\\#', '#')
    text = text.replace('\\_', '_')
    text = text.replace('\\{', '{')
    text = text.replace('\\}', '}')
    text = text.replace('\\textbullet', '')
    text = text.replace('\\par', ' ')
    text = text.replace('\\noindent', ' ')
    text = text.replace('\\hfill', ' ')
    text = text.replace('\\vspace{.*?}', ' ')
    text = text.replace('\\hspace{.*?}', ' ')
    text = text.replace('\\centering', ' ')
    text = text.replace('\\raggedright', ' ')
    text = text.replace('\\hline', ' ')
    text = text.replace('\\arraystretch{.*?}', ' ')
    text = text.replace('\\documentclass{.*?}', ' ')
    text = text.replace('\\usepackage{.*?}', ' ')
    text = text.replace('\\begin{document}', ' ')
    text = text.replace('\\end{document}', ' ')
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+\d+$', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[\.,;:!?]{2,}', '', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    logger.debug(f"Cleaned text (first 100 chars, original len {initial_len}): {text[:100]}...")
    return text

def calculate_text_quality_score(text):
    if not text:
        return 0.0
    words = word_tokenize(text)
    if not words:
        return 0.0
    ENGLISH_WORDS = set(nltk.corpus.words.words())
    english_word_count = sum(1 for word in words if word.lower() in ENGLISH_WORDS)
    english_word_ratio = english_word_count / len(words) if words else 0
    sentences = sent_tokenize(text)
    well_formed_sentences = sum(1 for sent in sentences if sent.strip().endswith(('.', '!', '?')))
    sentence_structure_score = well_formed_sentences / len(sentences) if sentences else 0
    quality_score = (english_word_ratio * 0.7) + (sentence_structure_score * 0.3)
    logger.debug(f"Text quality score calculated: {quality_score} for text (first 50 chars): {text[:50]}...")
    return quality_score

def is_garbage(text, threshold=GARBAGE_THRESHOLD, lenword=LENWORD):
    logger.debug(f"Checking if text is garbage (first 100 chars): {text[:100]}...")
    if not text or len(text.split()) < 5:
        logger.debug("Identified as garbage: text too short or empty.")
        return True
    try:
        if detect(text) != 'en':
            logger.debug("Identified as garbage: language not English.")
            return True
    except LangDetectException as e:
        logger.debug(f"Language detection failed for text (first 50 chars): {text[:50]}... Error: {e}. Assuming garbage.")
        return True
    words_list = text.split()
    for word in words_list:
        if len(word) > lenword and not '-' in word:
             logger.debug(f"Identified as garbage: found jammed word '{word[:50]}...'")
             return True
    quality_score = calculate_text_quality_score(text)
    if quality_score < threshold:
        logger.debug(f"Identified as garbage: quality score {quality_score} below threshold {threshold}.")
        return True
    logger.debug("Text passed garbage checks.")
    return False

def chunk_text(content, max_size=8192):
    logger.debug(f"Starting chunking process with max_size={max_size}.")
    segments = []
    current_segment = ""
    lines = content.split('\n')
    logger.debug(f"Splitting content into {len(lines)} lines.")
    for i, line in enumerate(lines):
        if line.strip().startswith(("# ", "## ", "### ")):
            logger.debug(f"Found markdown heading at line {i}: {line.strip()}")
            if current_segment:
                logger.debug(f"Processing previous segment before heading (length: {len(current_segment)}).")
                segments.extend(split_segment(current_segment.strip(), max_size))
            current_segment = line + "\n"
            logger.debug("Starting new segment after heading.")
        else:
            current_segment += line + "\n"
    if current_segment:
        logger.debug(f"Processing final segment (length: {len(current_segment)}).")
        segments.extend(split_segment(current_segment.strip(), max_size))
    logger.info(f"Chunking complete. Produced {len(segments)} initial segments based on headings.")
    return segments

def split_segment(segment, max_size):
    logger.debug(f"Splitting segment by sentences (length: {len(segment)}).")
    sentences = sent_tokenize(segment)
    logger.debug(f"Segment split into {len(sentences)} sentences.")
    chunks = []
    current_chunk = ""
    for i, sentence in enumerate(sentences):
        sentence_to_add = sentence + " " if current_chunk else sentence
        if len(current_chunk) + len(sentence_to_add) <= max_size:
            current_chunk += sentence_to_add
            logger.debug(f"Added sentence {i+1}/{len(sentences)} to current chunk (current size: {len(current_chunk)}).")
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                logger.debug(f"Chunk completed (size: {len(current_chunk)}). Starting new chunk with sentence {i+1}.")
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
        logger.debug(f"Added final chunk (size: {len(current_chunk)}).")
    logger.debug(f"Segment split into {len(chunks)} smaller chunks.")
    return chunks

def process_and_chunk_mmd(mmd_path, output_dir):
    logger.info(f"Starting processing and chunking for MMD file: {mmd_path}")
    if not mmd_path or not os.path.exists(mmd_path):
        logger.warning(f"MMD file not found or path is invalid: {mmd_path}. Skipping processing and chunking.")
        return None, None
    sanitized_filename = sanitize_filename(os.path.basename(mmd_path))
    cleaned_jsonl_path = os.path.join(output_dir, f"{os.path.splitext(sanitized_filename)[0]}_cleaned.jsonl")
    garbage_jsonl_path = os.path.join(output_dir, f"{os.path.splitext(sanitized_filename)[0]}_garbage.jsonl")
    try:
        with open(mmd_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.debug(f"Successfully read content from {mmd_path} (length: {len(content)}).")
    except Exception as e:
        logger.error(f"Error reading MMD file {mmd_path}: {e}")
        return None, None
    chunks = chunk_text(content)
    logger.info(f"MMD content chunked into {len(chunks)} segments.")
    cleaned_count = 0
    garbage_count = 0
    try:
        with open(cleaned_jsonl_path, 'w', encoding='utf-8') as cleaned_f, \
             open(garbage_jsonl_path, 'w', encoding='utf-8') as garbage_f:
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)} (length: {len(chunk)}).")
                cleaned_chunk = clean_text(chunk)
                if is_garbage(cleaned_chunk):
                    garbage_f.write(json.dumps({"text": cleaned_chunk}) + '\n')
                    garbage_count += 1
                    logger.debug(f"Chunk {i+1} identified as garbage.")
                else:
                    cleaned_f.write(json.dumps({"text": cleaned_chunk}) + '\n')
                    cleaned_count += 1
                    logger.debug(f"Chunk {i+1} identified as cleaned text.")
        logger.info(f"Finished processing and chunking {mmd_path}. Generated {cleaned_count} cleaned chunks and {garbage_count} garbage chunks.")
        return cleaned_jsonl_path, garbage_jsonl_path
    except Exception as e:
        logger.error(f"Error during cleaning or writing chunk files for {mmd_path}: {e}")
        if os.path.exists(cleaned_jsonl_path):
            os.remove(cleaned_jsonl_path)
        if os.path.exists(garbage_jsonl_path):
            os.remove(garbage_jsonl_path)
        return None, None

def upload_to_gcs(file_path, bucket_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(file_path)

    logger.info(f"File {file_path} uploaded to {destination_blob_name}.")

def process_single_rar(rar_file_url, bucket_name):
    """Processes a single RAR file: downloads, extracts, processes PDFs, and uploads to GCS."""
    rar_filename = rar_file_url.split('/')[-1]
    sanitized_rar_filename = sanitize_filename(rar_filename)
    rar_path = sanitized_rar_filename
    extract_path = os.path.splitext(rar_path)[0]

    logger.info(f"--- Processing {rar_filename} ---")

    if not download_file(rar_file_url, rar_path):
        logger.error(f"Failed to download RAR file: {rar_file_url}. Skipping.")
        return 0

    if not extract_rar(rar_path, extract_path):
        logger.error(f"Failed to extract RAR file: {rar_path}. Cleaning up and skipping.")
        if os.path.exists(rar_path):
            os.remove(rar_path)
        return 0

    if os.path.exists(rar_path):
        os.remove(rar_path)

    pdf_files = [os.path.join(root, file) for root, _, files in os.walk(extract_path) for file in files if file.lower().endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files in extracted directory: {extract_path}")

    if not pdf_files:
        logger.warning(f"No PDF files found in {extract_path}. Cleaning up.")
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        return 0

    successful_uploads_count = 0
    for pdf_path in pdf_files:
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Check if the cleaned JSONL file already exists in GCS
        pdf_basename = os.path.basename(pdf_path)
        cleaned_jsonl_name = f"{os.path.splitext(pdf_basename)[0]}_cleaned.jsonl"
        garbage_jsonl_name = f"{os.path.splitext(pdf_basename)[0]}_garbage.jsonl"
        
        # Check if cleaned file already exists in GCS
        if check_gcs_file_exists(bucket_name, f"cleaned/{cleaned_jsonl_name}"):
            logger.info(f"Cleaned JSONL file for {pdf_basename} already exists in GCS. Skipping processing.")
            continue
        
        # Check if garbage file already exists in GCS
        if check_gcs_file_exists(bucket_name, f"garbage/{garbage_jsonl_name}"):
            logger.info(f"Garbage JSONL file for {pdf_basename} already exists in GCS. Skipping processing.")
            continue
        
        # Process the PDF if it hasn't been processed yet
        mmd_path = process_pdf(pdf_path, extract_path)
        if mmd_path:
            logger.info(f"Nougat processing successful for {pdf_path}. MMD file: {mmd_path}")
            cleaned_jsonl, garbage_jsonl = process_and_chunk_mmd(mmd_path, extract_path)
            if cleaned_jsonl and os.path.exists(cleaned_jsonl):
                destination_blob_name = f"cleaned/{os.path.basename(cleaned_jsonl)}"
                upload_to_gcs(cleaned_jsonl, bucket_name, destination_blob_name)
                
                # Upload cleaned JSONL to Hugging Face
                upload_to_huggingface(cleaned_jsonl, HUGGING_FACE_REPO)
                
                successful_uploads_count += 1
            if garbage_jsonl and os.path.exists(garbage_jsonl):
                destination_blob_name = f"garbage/{os.path.basename(garbage_jsonl)}"
                upload_to_gcs(garbage_jsonl, bucket_name, destination_blob_name)
        else:
            logger.error(f"Nougat processing failed for {pdf_path}. Skipping cleaning, chunking, and upload.")

    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)

    return successful_uploads_count


def main():
    """Main function to process the library."""
    logger.info("--- Starting Library Processing Pipeline ---")

    if not BUCKET_NAME:
        logger.error("BUCKET_NAME environment variable not set. Exiting.")
        return

    # Get the list of RAR files to process
    rar_files = get_file_list(BASE_URL)

    # Distribute the work among the pods
    pod_index = int(os.environ.get("JOB_COMPLETION_INDEX", 0))
    num_pods = int(os.environ.get("NUM_PODS", 1))
    files_per_pod = len(rar_files) // num_pods
    start_index = pod_index * files_per_pod
    end_index = start_index + files_per_pod

    if pod_index == num_pods - 1:
        end_index = len(rar_files)

    files_to_process = rar_files[start_index:end_index]

    for rar_file_url in files_to_process:
        process_single_rar(rar_file_url, BUCKET_NAME)

    logger.info("--- Library Processing Pipeline Finished ---")

if __name__ == "__main__":
    main()
