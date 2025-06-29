import ollama
import os
import json
import sys
import base64
import io
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "qwen2.5vl:72b-q4_K_M"
PROMPT = "Extract all text from this page, preserving the structure (including tables and graphs). Output each sentence on a new line."
# ---------------------

def get_image_b64(image: Image.Image) -> str:
    """Converts a Pillow image to a base64 encoded string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_image(client: ollama.Client, file_path: Path, pbar: tqdm) -> list[dict]:
    """Opens an image, sends it to the model, and returns a list with a single page result."""
    try:
        pbar.write(f"Processing image: {file_path.name}")
        with Image.open(file_path) as img:
            img_b64 = get_image_b64(img.convert('RGB'))
            response = client.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': PROMPT, 'images': [img_b64]}]
            )
            pbar.write(f"Finished processing image: {file_path.name}")
            # Return as a list to be consistent with process_pdf
            return [{'page_number': 1, 'text': response['message']['content']}]
    except Exception as e:
        error_message = f"Error processing image {file_path}: {e}"
        pbar.write(error_message)
        return [{'page_number': 1, 'text': error_message}]

def process_pdf(client: ollama.Client, file_path: Path, pbar: tqdm) -> list[dict]:
    """Converts a PDF to images, sends each page to the model, and returns a list of page results."""
    page_results = []
    try:
        pbar.write(f"Converting {file_path.name} to images...")
        images = convert_from_path(file_path)
        pbar.write(f"Found {len(images)} pages in {file_path.name}.")

        page_iterator = tqdm(
            enumerate(images),
            total=len(images),
            desc=f"OCR on {file_path.name}",
            unit="page",
            leave=False
        )
        for i, image in page_iterator:
            img_b64 = get_image_b64(image.convert('RGB'))
            response = client.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': f"{PROMPT} (Page {i+1}/{len(images)})", 'images': [img_b64]}]
            )
            page_results.append({'page_number': i + 1, 'text': response['message']['content']})

        pbar.write(f"Finished processing {file_path.name}.")
    except Exception as e:
        error_message = f"Error processing PDF {file_path}: {e}"
        pbar.write(error_message)
        # Add error as a result for this page to be logged
        page_results.append({'page_number': 0, 'text': error_message})
    return page_results

def get_file_handler(file_extension: str):
    """Returns the appropriate handler function for a given file extension."""
    handlers = {
        '.pdf': process_pdf,
        '.png': process_image,
        '.jpg': process_image,
        '.jpeg': process_image,
        '.bmp': process_image,
        '.gif': process_image,
    }
    return handlers.get(file_extension.lower())

def discover_files(input_dir: Path):
    """Recursively discovers all files in the input directory."""
    return [p for p in input_dir.rglob("*") if p.is_file()]

def save_to_jsonl(output_file: Path, data: dict):
    """Saves a dictionary to a JSONL file."""
    with output_file.open('w', encoding='utf-8') as f:
        json.dump(data, f)
        f.write('\n')

def main(input_dir: Path, output_dir: Path):
    """Main function to walk through a directory and process files."""
    client = ollama.Client()
    try:
        client.show(MODEL_NAME)
    except ollama.ResponseError as e:
        print(f"Error: Model '{MODEL_NAME}' not found. Please ensure it's available in Ollama.", file=sys.stderr)
        print(f"Details: {e.error}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(exist_ok=True)
    print(f"Output will be saved in: {output_dir.resolve()}")

    files_to_process = discover_files(input_dir)
    
    with tqdm(total=len(files_to_process), desc="Overall Progress", unit="file") as pbar:
        for file_path in files_to_process:
            pbar.set_description(f"Processing {file_path.name}")
            
            handler = get_file_handler(file_path.suffix)
            if not handler:
                pbar.update(1)
                continue

            page_results = handler(client, file_path, pbar)

            if page_results:
                for page_data in page_results:
                    page_num = page_data['page_number']
                    text = page_data['text']
                    
                    # For multi-page docs or PDFs, create a per-page file
                    if len(page_results) > 1 or file_path.suffix.lower() == '.pdf':
                         output_filename = f"{file_path.name}.page_{page_num}.jsonl"
                    else: # For single images
                         output_filename = f"{file_path.name}.jsonl"

                    output_filepath = output_dir / output_filename
                    
                    json_data = {
                        'source_file': str(file_path),
                        'page_number': page_num,
                        'extracted_text': text
                    }
                    save_to_jsonl(output_filepath, json_data)
            
            pbar.update(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_text.py <input_directory> [output_directory]", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("extracted_output")

    if not input_path.is_dir():
        print(f"Error: Input directory '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    main(input_path, output_path)