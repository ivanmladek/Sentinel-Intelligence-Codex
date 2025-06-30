import os
import sys
import subprocess
from pathlib import Path
from tqdm import tqdm

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



def discover_files(input_dir: Path):
    """Recursively discovers all PDF files in the input directory."""
    return list(input_dir.rglob("*.pdf"))

def main(input_dir: Path, output_dir: Path):
    output_dir.mkdir(exist_ok=True)
    print(f"Output will be saved in: {output_dir.resolve()}")
    

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

    with tqdm(total=len(files_to_process), desc="Overall Progress", unit="file") as pbar:
        for file_path in files_to_process:
            result = ocr_with_nougat(file_path, output_dir)
            pbar.write(result)
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
