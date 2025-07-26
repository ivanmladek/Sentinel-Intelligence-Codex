# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    unzip \
    unrar \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Nougat OCR
RUN pip install --no-deps git+https://github.com/facebookresearch/nougat

# Install XLA support (optional)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir https://storage.googleapis.com/tpu-pytorch/wheels-cu118/torch_xla-2.0-cp39-cp39-linux_x86_64.whl

# Install Triton support (optional)
RUN pip install --no-cache-dir triton

# Install NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('words'); nltk.download('punkt_tab')"

# Copy the distributed processing scripts into the container
COPY distributed_pdf_processor_parallel.py .
COPY distributed_extract_text.py .

# Make sure the scripts are executable
RUN chmod +x distributed_pdf_processor_parallel.py
RUN chmod +x distributed_extract_text.py

# Set environment variables
ENV PROJECT_ID=gdrive-410709
ENV REGION=us-central1
ENV GCS_BUCKET_NAME=gdrive-410709-vllm-13b-demo-bucket

# Expose the port the app runs on (if needed)
EXPOSE 8080

# Define the command to run the distributed PDF processor
ENTRYPOINT ["python", "distributed_pdf_processor_parallel.py"]