import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

def ocr_with_nougat(file_path: Path, output_dir: Path, pbar: tqdm):
    """Performs OCR on a PDF using the Nougat command-line tool and logs a sample of the output."""
    try:
        pbar.write(f"Running Nougat OCR for {file_path.name}...")
        
        # Nougat saves the output file with the same stem as the input, but with a .mmd extension.
        expected_output_path = output_dir / f"{file_path.stem}.mmd"

        # Get the full path to the nougat script
        nougat_script_path = "/Users/jj/.pyenv/versions/3.11.4/bin/nougat"
        python_interpreter_path = "/Users/jj/.pyenv/versions/3.11.4/bin/python"

        # Run the nougat command
        command = [
            python_interpreter_path,
            nougat_script_path,
            str(file_path),
            "-o",
            str(output_dir)
        ]
        
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        
        if process.returncode != 0:
            pbar.write(f"  [Error] Nougat process failed for {file_path.name}:")
            pbar.write(f"  {process.stderr}")
            return

        # Verify that the output file was created
        if expected_output_path.exists():
            with expected_output_path.open('r', encoding='utf-8') as f:
                sample_text = f.read(100).replace('\n', ' ')
            pbar.write(f"  [Success] Nougat created '{expected_output_path.name}'. Sample: {sample_text}...")
        else:
            pbar.write(f"  [Error] Nougat finished but the output file '{expected_output_path.name}' was not found.")
            
    except Exception as e:
        pbar.write(f"  [Error] An exception occurred during Nougat OCR for {file_path.name}: {e}")

def discover_files(input_dir: Path):
    """Recursively discovers all PDF files in the input directory."""
    return list(input_dir.rglob("*.pdf"))

def main(input_dir: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    print(f"Output will be saved in: {output_dir.resolve()}")
    print("Processing all PDF files with Nougat.")

    files_to_process = discover_files(input_dir)
    
    if not files_to_process:
        print("No PDF files found in the input directory.")
        return

    with tqdm(total=len(files_to_process), desc="Overall Progress", unit="file") as pbar:
        for file_path in files_to_process:
            pbar.set_description(f"Processing {file_path.name}")
            ocr_with_nougat(file_path, output_dir, pbar)
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