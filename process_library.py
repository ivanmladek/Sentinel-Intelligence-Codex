
import os
import re
import json
import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import fcntl
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
from transformers import pipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://the-eye.eu/public/Books/Bibliotheca%20Alexandrina/"
CSV_FILE = "world2.csv"
HUGGING_FACE_REPO = "ivanmladek/Sentinel-Intelligence-Codex"  # Replace with your Hugging Face repo
GARBAGE_THRESHOLD = 0.8
LENWORD = 50

def get_file_list(url):
    """Get a list of files from a URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [link.get('href') for link in soup.find_all('a') if link.get('href').endswith('.rar')]

def download_file(url, output_path):
    """Download a file from a URL."""
    if os.path.exists(output_path):
        logger.info(f"{output_path} already exists. Skipping download.")
        return
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def extract_rar(file_path, output_path):
    """Extract a RAR file."""
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    try:
        subprocess.run(['unrar', 'x', file_path, output_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting {file_path}: {e}")


def sanitize_filename(filename):
    """Sanitize a filename."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

def process_pdf(pdf_path, output_dir):
    """Process a single PDF file with Nougat."""
    sanitized_filename = sanitize_filename(os.path.basename(pdf_path))
    mmd_path = os.path.join(output_dir, f"{os.path.splitext(sanitized_filename)[0]}.mmd")
    if os.path.exists(mmd_path):
        logger.info(f"{mmd_path} already exists. Skipping Nougat processing.")
        return mmd_path

    try:
        subprocess.run(['nougat', pdf_path, '-o', output_dir, '--no-skipping', '--recompute'], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {pdf_path} with Nougat: {e}")
        return None
    return mmd_path


def clean_text(text):
    """Clean the extracted text."""
    text = re.sub(r'\\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    text = re.sub(r'\\[[^\\]]*\\]', '', text)
    text = re.sub(r'\\(\\d+\\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\[[A-Za-z0-9]+\\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\([\\w\\s]+et\\s+al\\., \\d{4}\\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\(\\w+\\s+and\\s+\\w+\\s+\\d{4}\\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\(see\\s+equations\\s+\\(\\d+\\)\\s+and\\s+\\(\\d+\\)\\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\(\\w+\\s+et\\s+al\\., \\d{4};\\s*\\w+\\s+et\\s+al\\., \\d{4}\\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Table\\s+\\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\[FIGURE:[^]]+\\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\[\\d+(,\\s*\\d+)*\\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\[.*arxiv.*\\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\\x00-\\x7F]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\\.,;:!?]{2,}', '', text, flags=re.IGNORECASE)
    return text


def is_garbage(text, threshold=GARBAGE_THRESHOLD, lenword=LENWORD):
    """Check if the text is garbage."""
    # This is a simplified version of the garbage detection from the notebook.
    # A more robust implementation would be needed for production.
    if not text or len(text.split()) < 10:
        return True
    try:
        if detect(text) != 'en':
            return True
    except LangDetectException:
        return True
    return False

def chunk_text(content, max_size=8192):
    """Chunk the text into smaller segments."""
    segments = []
    current_segment = ""
    lines = content.split('\\n')

    for line in lines:
        if line.startswith(("#", "##", "###")):
            if current_segment:
                segments.extend(split_segment(current_segment.strip(), max_size))
            current_segment = ""
        else:
            current_segment += line + " "

    if current_segment:
        segments.extend(split_segment(current_segment.strip(), max_size))

    return segments

def split_segment(segment, max_size):
    """Split a segment into smaller chunks."""
    sentences = sent_tokenize(segment)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def process_and_chunk_mmd(mmd_path, output_dir):
    """Process and chunk an MMD file."""
    if not mmd_path or not os.path.exists(mmd_path):
        logger.warning(f"MMD file not found: {mmd_path}. Skipping.")
        return None, None

    cleaned_jsonl_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(mmd_path))[0]}_cleaned.jsonl")
    garbage_jsonl_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(mmd_path))[0]}_garbage.jsonl")

    with open(mmd_path, 'r', encoding='utf-8') as f:
        content = f.read()

    chunks = chunk_text(content)
    with open(cleaned_jsonl_path, 'w') as cleaned_f, open(garbage_jsonl_path, 'w') as garbage_f:
        for chunk in chunks:
            cleaned_chunk = clean_text(chunk)
            if is_garbage(cleaned_chunk):
                garbage_f.write(json.dumps({"text": cleaned_chunk}) + '\\n')
            else:
                cleaned_f.write(json.dumps({"text": cleaned_chunk}) + '\\n')

    return cleaned_jsonl_path, garbage_jsonl_path


def upload_to_huggingface(file_path, repo_id):
    """Upload a file to a Hugging Face repository."""
    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),
        repo_id=repo_id,
        repo_type="dataset",
    )
    logger.info(f"Uploaded {file_path} to {repo_id}")


def main():
    """Main function to process the library."""
    # Download necessary NLTK data
    nltk.download('punkt')
    nltk.download('words')

    # Get the list of all books from the CSV
    all_books = pd.read_csv(CSV_FILE)['Book'].tolist()
    total_books = len(all_books)
    processed_books = 0

    # Get the list of RAR files from the URL
    rar_files = get_file_list(BASE_URL)

    with tqdm(total=total_books, desc="Processing Library") as pbar:
        for rar_file in rar_files:
            rar_url = BASE_URL + rar_file
            rar_path = sanitize_filename(rar_file)
            extract_path = os.path.splitext(rar_path)[0]

            # Download and extract the RAR file
            download_file(rar_url, rar_path)
            extract_rar(rar_path, extract_path)

            # Process each PDF in the extracted directory
            for root, _, files in os.walk(extract_path):
                for file in files:
                    if file.endswith('.pdf'):
                        pdf_path = os.path.join(root, file)
                        mmd_path = process_pdf(pdf_path, extract_path)
                        cleaned_jsonl, garbage_jsonl = process_and_chunk_mmd(mmd_path, extract_path)

                        # Upload to Hugging Face
                        if cleaned_jsonl and os.path.exists(cleaned_jsonl):
                            upload_to_huggingface(cleaned_jsonl, HUGGING_FACE_REPO)
                        if garbage_jsonl and os.path.exists(garbage_jsonl):
                            upload_to_huggingface(garbage_jsonl, HUGGING_FACE_REPO)

                        # Update progress
                        book_title = os.path.splitext(os.path.basename(pdf_path))[0]
                        if book_title in all_books:
                            processed_books += 1
                            pbar.update(1)
                            pbar.set_postfix({"Processed": f"{processed_books}/{total_books}"})

            # Clean up
            os.remove(rar_path)
            shutil.rmtree(extract_path)

if __name__ == "__main__":
    main()
