#!/usr/bin/env python3
"""
Test script for the distributed PDF processing pipeline
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the current directory to the Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from distributed_pdf_processor_parallel import main

def create_test_pdf_files(test_dir: str, num_files: int = 3) -> str:
    """
    Create test PDF files for testing the pipeline.
    
    In a real test, we would use actual PDF files, but for this test we'll
    create simple text files with a .pdf extension to simulate PDFs.
    """
    test_input_dir = Path(test_dir) / "input_pdfs"
    test_input_dir.mkdir(exist_ok=True)
    
    for i in range(num_files):
        pdf_file = test_input_dir / f"test_document_{i+1}.pdf"
        with open(pdf_file, "w") as f:
            f.write(f"This is test PDF document #{i+1}\n")
            f.write("This file is used to test the distributed PDF processing pipeline.\n")
            f.write("In a real scenario, this would be an actual PDF file.\n")
    
    return str(test_input_dir)

def test_distributed_pipeline():
    """Test the distributed PDF processing pipeline."""
    print("Testing distributed PDF processing pipeline...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create test PDF files
        input_dir = create_test_pdf_files(temp_dir, 3)
        output_dir = str(Path(temp_dir) / "output")
        
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        # Run the distributed pipeline
        try:
            main(input_dir, output_dir, max_workers=2)
            print("Pipeline completed successfully!")
        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            return False
        
        # Check if output files were created
        output_path = Path(output_dir)
        if not output_path.exists():
            print("Output directory was not created!")
            return False
        
        output_files = list(output_path.glob("*.mmd"))
        print(f"Found {len(output_files)} output files:")
        for output_file in output_files:
            print(f"  - {output_file.name}")
        
        # Check if we have the expected number of output files
        if len(output_files) == 0:
            print("No output files were created!")
            return False
        
        print("Test completed successfully!")
        return True

if __name__ == "__main__":
    success = test_distributed_pipeline()
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Tests failed!")
        sys.exit(1)