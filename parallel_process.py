import os
import re
import json
import logging
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import nltk
import requests
from bs4 import BeautifulSoup
from huggingface_hub import HfApi
from langdetect import detect, LangDetectException
from nltk.corpus import words, brown
from nltk.tokenize import word_tokenize, sent_tokenize
from google.cloud import storage

# Config
BASE_URL = os.environ.get("BASE_URL", "https://the-eye.eu/public/Books/Bibliotheca%20Alexandrina/")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
HUGGING_FACE_REPO = os.environ.get("HUGGING_FACE_REPO", "Disperser5601/Sentinel-Intelligence-Codex")
GARBAGE_THRESHOLD = 0.5
LENWORD = 50

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
nltk.download(['punkt', 'words', 'punkt_tab'], quiet=True)
ENGLISH_WORDS = set(nltk.corpus.words.words())

def get_file_list(url, depth=0, max_depth=3):
    """Recursively get RAR files from URL."""
    if depth > max_depth:
        return []

    rar_files = []
    adapter = requests.adapters.HTTPAdapter(max_retries=requests.packages.urllib3.util.retry.Retry(
        total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]))
    http = requests.Session()
    http.mount("http://", adapter)
    http.mount("https://", adapter)

    try:
        response = http.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for link in soup.find_all('a'):
            href = link.get('href')
            if href:
                full_url = requests.compat.urljoin(url, href)
                if full_url.startswith(url) and len(full_url) > len(url) and full_url.endswith('/'):
                    rar_files.extend(get_file_list(full_url, depth + 1, max_depth))
                elif full_url.endswith('.rar'):
                    rar_files.append(full_url)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error accessing {url}: {e}")

    return rar_files

def download_file(url, output_path):
    """Download file if it doesn't exist."""
    if os.path.exists(output_path):
        return True
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed {url}: {e}")
        return False

def extract_rar(file_path, output_path):
    """Extract RAR file."""
    if not os.path.exists(file_path):
        return False
    os.makedirs(output_path, exist_ok=True)
    try:
        subprocess.run(['unrar', 'x', '-o+', file_path, output_path], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"Extraction failed {file_path}: {e}")
        return False

def sanitize_filename(filename):
    """Sanitize filename."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

def process_pdf(pdf_path, output_dir):
    """Process PDF with Nougat."""
    if not os.path.exists(pdf_path):
        return None

    os.makedirs(output_dir, exist_ok=True)
    mmd_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.mmd")

    if os.path.exists(mmd_path):
        return mmd_path

    try:
        process = subprocess.Popen(
            ['nougat', pdf_path, '-o', output_dir, '--batchsize', '2', '--no-skipping'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)

        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        if process.wait() == 0 and os.path.exists(mmd_path):
            return mmd_path
    except Exception as e:
        logger.error(f"Nougat failed {pdf_path}: {e}")

    return None

def check_gcs_file_exists(bucket_name, blob_name):
    """Check if file exists in GCS."""
    try:
        storage_client = storage.Client()
        return storage_client.bucket(bucket_name).blob(blob_name).exists()
    except Exception as e:
        logger.error(f"GCS check failed {blob_name}: {e}")
        return False

def upload_to_huggingface(file_path, repo_id, repo_type="dataset"):
    """Upload to Hugging Face."""
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token or not os.path.exists(file_path):
        return False

    try:
        api = HfApi(token=hf_token)
        api.upload_file(path_or_fileobj=file_path, path_in_repo=os.path.basename(file_path),
                       repo_id=repo_id, repo_type=repo_type)
        return True
    except Exception as e:
        logger.error(f"HF upload failed {file_path}: {e}")
        return False

def check_huggingface_file_exists(repo_id, filename, repo_type="dataset"):
    """Check if file exists on HF."""
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")
    if not hf_token:
        return False

    try:
        api = HfApi(token=hf_token)
        return filename in api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except Exception as e:
        logger.error(f"HF check failed {filename}: {e}")
        return False

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""

    # Remove markdown headers, formatting, and LaTeX
    text = re.sub(r'^\s*#+\s+.*', '', text, flags=re.MULTILINE)
    text = re.sub(r'(\*|_){1,2}(.*?)\1', r'\2', text)
    text = re.sub(r'\\[a-zA-Z]+(\{.*?\})?', '', text)
    text = re.sub(r'\$.*?\$', '', text)

    # Remove brackets, tables, figures
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    text = re.sub(r'(?i)(table|figure)\s+\d+.*?(?=\n|$)', '', text)

    # Normalize whitespace and remove non-ASCII
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'[\.,;:!?]{2,}', '.', text)

    return text.strip()

def calculate_text_quality_score(text):
    """Calculate text quality score."""
    if not text:
        return 0.0

    words = word_tokenize(text)
    if not words:
        return 0.0

    english_ratio = sum(1 for word in words if word.lower() in ENGLISH_WORDS) / len(words)
    sentences = sent_tokenize(text)
    sentence_score = sum(1 for sent in sentences if sent.strip().endswith(('.', '!', '?'))) / len(sentences) if sentences else 0

    return (english_ratio * 0.7) + (sentence_score * 0.3)

def is_garbage(text, threshold=GARBAGE_THRESHOLD, lenword=LENWORD):
    """Check if text is garbage."""
    if not text or len(text.split()) < 5:
        return True

    try:
        if detect(text) != 'en':
            return True
    except LangDetectException:
        return True

    if any(len(word) > lenword and '-' not in word for word in text.split()):
        return True

    return calculate_text_quality_score(text) < threshold

def chunk_text(content, max_size=8192):
    """Chunk text by headings and sentence boundaries."""
    segments = []
    current_segment = ""

    for line in content.split('\n'):
        if line.strip().startswith(("# ", "## ", "### ")):
            if current_segment:
                segments.extend(split_segment(current_segment.strip(), max_size))
            current_segment = line + "\n"
        else:
            current_segment += line + "\n"

    if current_segment:
        segments.extend(split_segment(current_segment.strip(), max_size))

    return segments

def split_segment(segment, max_size):
    """Split segment by sentences."""
    sentences = sent_tokenize(segment)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence_to_add = sentence + " " if current_chunk else sentence
        if len(current_chunk) + len(sentence_to_add) <= max_size:
            current_chunk += sentence_to_add
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def process_and_chunk_mmd(mmd_path, output_dir):
    """Process MMD file and create cleaned/garbage chunks."""
    if not mmd_path or not os.path.exists(mmd_path):
        return None, None

    try:
        with open(mmd_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Error reading {mmd_path}: {e}")
        return None, None

    chunks = chunk_text(content)
    base_name = sanitize_filename(os.path.basename(mmd_path))
    cleaned_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_cleaned.jsonl")
    garbage_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_garbage.jsonl")

    try:
        with open(cleaned_path, 'w', encoding='utf-8') as cleaned_f, \
             open(garbage_path, 'w', encoding='utf-8') as garbage_f:

            for chunk in chunks:
                cleaned_chunk = clean_text(chunk)
                target_file = garbage_f if is_garbage(cleaned_chunk) else cleaned_f
                target_file.write(json.dumps({"text": cleaned_chunk}) + '\n')

        return cleaned_path, garbage_path
    except Exception as e:
        logger.error(f"Error processing {mmd_path}: {e}")
        for path in [cleaned_path, garbage_path]:
            if os.path.exists(path):
                os.remove(path)
        return None, None

def upload_to_gcs(file_path, bucket_name, destination_blob_name):
    """Upload file to GCS."""
    try:
        storage_client = storage.Client()
        storage_client.bucket(bucket_name).blob(destination_blob_name).upload_from_filename(file_path)
        logger.info(f"Uploaded {file_path} to {destination_blob_name}")
    except Exception as e:
        logger.error(f"GCS upload failed {file_path}: {e}")

def cleanup_files(*file_paths):
    """Clean up files to save space."""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path) if os.path.isfile(path) else shutil.rmtree(path)
                logger.info(f"Cleaned up {path}")
            except Exception as e:
                logger.error(f"Cleanup failed {path}: {e}")

def process_single_rar(rar_file_url, bucket_name):
    """Process single RAR file."""
    rar_filename = rar_file_url.split('/')[-1]
    rar_path = sanitize_filename(rar_filename)
    extract_path = os.path.splitext(rar_path)[0]

    logger.info(f"Processing {rar_filename}")

    # Skip if RAR already processed
    if check_gcs_file_exists(bucket_name, f"rars/{rar_filename}"):
        cleanup_files(rar_path)
        return 0

    # Download and extract
    if not download_file(rar_file_url, rar_path) or not extract_rar(rar_path, extract_path):
        cleanup_files(rar_path)
        return 0

    cleanup_files(rar_path)  # Remove RAR after extraction

    # Find PDFs
    pdf_files = [os.path.join(root, file) for root, _, files in os.walk(extract_path)
                 for file in files if file.lower().endswith('.pdf')]
    random.shuffle(pdf_files)

    if not pdf_files:
        cleanup_files(extract_path)
        return 0

    successful_uploads = 0

    for pdf_path in pdf_files:
        pdf_basename = os.path.basename(pdf_path)
        cleaned_name = f"{os.path.splitext(pdf_basename)[0]}_cleaned.jsonl"
        garbage_name = f"{os.path.splitext(pdf_basename)[0]}_garbage.jsonl"

        # Skip if already processed
        if (check_gcs_file_exists(bucket_name, f"cleaned/{cleaned_name}") or
            check_huggingface_file_exists(HUGGING_FACE_REPO, cleaned_name) or
            check_gcs_file_exists(bucket_name, f"garbage/{garbage_name}") or
            check_huggingface_file_exists(HUGGING_FACE_REPO, garbage_name)):
            cleanup_files(pdf_path)
            continue

        # Upload PDF to GCS if not exists
        if not check_gcs_file_exists(bucket_name, f"pdfs/{pdf_basename}"):
            try:
                upload_to_gcs(pdf_path, bucket_name, f"pdfs/{pdf_basename}")
            except Exception as e:
                logger.error(f"PDF upload failed {pdf_basename}: {e}")

        # Process PDF
        mmd_path = process_pdf(pdf_path, extract_path)
        if mmd_path:
            cleaned_jsonl, garbage_jsonl = process_and_chunk_mmd(mmd_path, extract_path)

            # Upload cleaned file
            if cleaned_jsonl and os.path.exists(cleaned_jsonl):
                upload_to_gcs(cleaned_jsonl, bucket_name, f"cleaned/{os.path.basename(cleaned_jsonl)}")
                if not check_huggingface_file_exists(HUGGING_FACE_REPO, os.path.basename(cleaned_jsonl)):
                    upload_to_huggingface(cleaned_jsonl, HUGGING_FACE_REPO)
                successful_uploads += 1

            # Upload garbage file
            if garbage_jsonl and os.path.exists(garbage_jsonl):
                upload_to_gcs(garbage_jsonl, bucket_name, f"garbage/{os.path.basename(garbage_jsonl)}")

            cleanup_files(pdf_path, cleaned_jsonl, garbage_jsonl)
        else:
            cleanup_files(pdf_path)

    cleanup_files(extract_path)

    # Upload RAR to GCS
    if os.path.exists(rar_path):
        try:
            upload_to_gcs(rar_path, bucket_name, f"rars/{rar_filename}")
        except Exception as e:
            logger.error(f"RAR upload failed {rar_filename}: {e}")
        cleanup_files(rar_path)

    return successful_uploads

def main():
    """Main processing function."""
    logger.info("Starting Library Processing Pipeline")

    if not BUCKET_NAME:
        logger.error("BUCKET_NAME not set")
        return

    # Get RAR files
    rar_files = get_file_list(BASE_URL)
    random.shuffle(rar_files)

    # Distribute work among pods
    pod_index = int(os.environ["JOB_COMPLETION_INDEX"])
    num_pods = int(os.environ.get("NUM_PODS", 1))
    files_per_pod = len(rar_files) // num_pods
    start_index = pod_index * files_per_pod
    end_index = start_index + files_per_pod

    if pod_index == num_pods - 1:
        end_index = len(rar_files)

    for rar_file_url in rar_files[start_index:end_index]:
        process_single_rar(rar_file_url, BUCKET_NAME)

    logger.info("Library Processing Pipeline Finished")

if __name__ == "__main__":
    main()
