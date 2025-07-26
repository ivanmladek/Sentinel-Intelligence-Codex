# World History Book Collection Processing Pipeline

> [!NOTE]
> The books and PDFs referenced in this library are not hosted in this repository. They are freely available for download from the following public resource:
> 
> https://the-eye.eu/public/Books/Bibliotheca%20Alexandrina/

This project is a comprehensive pipeline for processing a curated collection of historical texts, offering a broad and deep exploration of human history from prehistory to the modern era. The pipeline is designed to extract, clean, and prepare text from PDF files for use in training large language models.

## Dataset Overview

The collection is organized into a comprehensive set of categories, with a significant emphasis on ancient and classical civilizations, as well as detailed accounts of various historical periods and regions.

The collection is particularly strong in its coverage of **Prehistory** and **Ancient & Classical Civilizations**. The prehistory section delves into archaeology, the Bronze Age, the Iron Age, and the Neolithic period, providing a rich foundation in the early chapters of human history. The section on ancient and classical civilizations is extensive, with in-depth collections on:

*   **Alexander the Great**: Covering his life, conquests, and the Hellenistic world that followed.
*   **Ancient Egypt**: A vast collection of texts on Egyptian dynasties, mythology, art, and archaeology.
*   **Ancient Greece**: Exploring the Archaic, Classical, and Hellenistic periods, with a focus on Athens, Sparta, and the broader Greek world.
*   **Ancient Rome**: A comprehensive look at the Roman Republic and Empire, including its military, society, and key historical figures.
*   **Other Civilizations**: The library also includes significant collections on the civilizations of the Near East, the Celts, the Vikings, and ancient China, among others.

Beyond the ancient world, the collection extends to **Medieval and Modern History**, with detailed sections on:

*   **The Middle Ages**: Covering the Byzantine Empire, the Crusades, and the development of European kingdoms.
*   **Early Modern History**: Exploring the Renaissance, the Reformation, and the Age of Discovery.
*   **Modern History**: With extensive material on World War I, World War II, the Cold War, and various regional conflicts.

The library is not limited to political and military history. It also includes substantial collections on:

*   **Art and Architecture**: With a focus on historical periods and styles.
*   **Philosophy and Religion**: Exploring the intellectual and spiritual traditions of various cultures.
*   **Science and Technology**: Charting the history of scientific discovery and technological innovation.

## Pipeline Architecture

The processing pipeline is designed to run on Google Cloud Platform (GCP) using Kubernetes (GKE) for scalable, parallel processing of the PDF files. The pipeline consists of the following components:

### 1. Terraform Orchestration

The infrastructure is managed using Terraform, which provisions the following resources:

*   **GKE Cluster**: A Google Kubernetes Engine cluster for running the processing jobs.
*   **GPU Node Pool**: A node pool with GPU instances for accelerated PDF processing.
*   **Cloud Storage Bucket**: A bucket for storing the processed text data.
*   **Artifact Registry**: A repository for storing the Docker images used in the pipeline.

### 2. GKE Cluster Setup

The GKE cluster is configured with the following specifications:

*   **Cluster Name**: `batch-inference-cluster`
*   **Zone**: `us-central1-b`
*   **Node Pool**: 
    * **Machine Type**: `n1-standard-4`
    * **Accelerator**: 4 x NVIDIA Tesla T4 GPUs
    * **Preemptible**: Enabled for cost optimization

### 3. Docker Containerization

The processing logic is packaged in a Docker container that includes all necessary dependencies:

*   **Base Image**: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04`
*   **Dependencies**: 
    * `nougat-ocr` for PDF text extraction
    * `nltk` for natural language processing
    * `langdetect` for language identification
    * Various other Python libraries for text processing

### 4. Parallel Processing with Kubernetes Jobs

The processing is orchestrated using Kubernetes Jobs that:

*   **Distribute Work**: Split the collection of PDF files across multiple pods for parallel processing
*   **Scale**: Run multiple instances of the processing container based on the available GPU resources
*   **Handle Failures**: Automatically restart failed pods to ensure completion of the processing

## Processing Script (`parallel_process.py`)

The core processing logic is implemented in `parallel_process.py`, which performs the following steps:

### 1. Environment Setup and PDF Discovery

- **Dependencies**: The script uses `nougat-ocr` for text extraction, `nltk` for natural language processing, and `langdetect` for language identification.
- **PDF Discovery**: The script recursively scans a specified URL to locate all RAR files containing PDFs.

### 2. Text Extraction with Nougat

- **Nougat OCR**: For each PDF, the `nougat` command-line tool is used. Nougat is a state-of-the-art OCR tool specifically designed for academic and scientific documents, capable of recognizing and transcribing complex layouts, mathematical equations, and tables into a structured Markdown format (`.mmd`).
- **Output**: The raw extracted text is saved as a `.mmd` file, preserving the document's structure with Markdown headings.

### 3. Text Cleaning and Garbage Detection

This is a critical step to filter out irrelevant or low-quality text.

- **Cleaning**: A series of regular expressions and cleaning functions are applied to the raw text to:
    - Remove extra newlines, spaces, and non-ASCII characters.
    - Eliminate academic citations, references to tables/figures, and bracketed content.
    - Sanitize garbled punctuation and symbols.
- **Garbage Detection**: Each segment of text is evaluated against a set of criteria to identify and discard "garbage" content. This includes:
    - **Language Detection**: Text that is not identified as English is discarded.
    - **Heuristics**: Checks for "jammed" words (long strings of characters without spaces), an unusually high proportion of single-letter words, and repetitive patterns.
    - **Quality Scoring**: A `text_quality_score` is calculated based on the presence of common English words, proper part-of-speech patterns, and other linguistic features. Text falling below a certain threshold is flagged as garbage.

### 4. Tokenization and Chunking

- **Chunking Strategy**: The cleaned `.mmd` content is chunked into smaller, manageable segments suitable for the LLM. The chunking logic is designed to respect the document's structure:
    - The text is split by Markdown headings (`#`, `##`, `###`).
    - These larger sections are then further divided into sentences using `nltk.sent_tokenize`.
- **Size Constraints**: The sentences are grouped into chunks with a maximum size (e.g., 8192 characters) to ensure they fit within the model's context window, while avoiding splitting sentences in the middle.
- **Final Output**: The cleaned, chunked text is saved to a `.jsonl` file, with each line containing a JSON object with a single "text" key, ready for training the LLM. Garbage text is saved to a separate file for review.

## Deployment

To deploy the pipeline:

1. **Provision Infrastructure**: Run `terraform apply` in the `terraform` directory to create the GKE cluster and associated resources.
2. **Build and Push Docker Image**: Build the processing container and push it to the Artifact Registry.
3. **Deploy Kubernetes Job**: Apply the `gke_job.yaml` manifest to start the parallel processing job.

The pipeline is designed to be scalable and cost-effective, leveraging GCP's preemptible GPU instances and Kubernetes' orchestration capabilities to process large collections of PDF files efficiently.