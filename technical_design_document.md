# **Section 1 - Project Description**

## **1.1 Project**

World History Book Collection Processing Pipeline (GCP)

## **1.2 Description**

This document outlines the technical design and architecture for the World History Book Collection Processing Pipeline, built entirely on Google Cloud Platform (GCP). The project's goal is to ingest a large collection of historical texts in PDF format, process them into a clean, structured format, and prepare them for ingestion by a Large Language Model (LLM). The source material is a vast and diverse collection of public domain books.

This pipeline is designed to be robust, scalable, and automated, leveraging GCP's powerful compute, storage, and orchestration services to transform unstructured PDFs into high-quality, structured JSONL data suitable for machine learning tasks. The infrastructure is defined as code using Terraform, as seen in the `terraform/main.tf` file.

## **1.3 Revision History**

| Date       | Comment                           | Author  |
| :--------- | :-------------------------------- | :------ |
| 2025-08-11 | Initial document creation         | Gemini  |
| 2025-08-11 | Refactored to match new template  | Gemini  |
| 2025-08-11 | Corrected list formatting         | Gemini  |
| 2025-08-11 | Clarified As-Is vs. To-Be Arch.   | Gemini  |

# **Section 2 - Overview**

## **2.1 Purpose**

The purpose of this project is to create a unique, high-quality dataset of historical texts for training and fine-tuning Large Language Models (LLMs) and to make a vast repository of historical knowledge accessible for AI-driven research and applications.

## **2.2 Scope**

The scope of this design covers the entire automated pipeline, including:

- **Automation:** A fully automated pipeline from PDF discovery to final JSONL output.
- **Scalability:** A system capable of processing tens of thousands of large PDF documents efficiently using a containerized workflow on GKE.
- **Accuracy:** Ensuring the highest possible fidelity in text extraction.
- **Data Quality:** Rigorous cleaning and filtering processes to remove OCR errors and irrelevant content.
- **Infrastructure as Code (IaC):** Management of all infrastructure components via Terraform.

## **2.3 Requirements**

This document does not track granular requirements. See Section 2.2 for the high-level scope and goals.

# **Section 3 - System Architecture**

This section describes the target (to-be) architecture for a fully automated, event-driven pipeline. The current infrastructure, as defined in `terraform/main.tf`, provisions the foundational components: GCS for storage, Artifact Registry for container images, and a GKE cluster for compute. 

The full data flow outlined below represents the proposed software architecture that will run on this infrastructure, utilizing additional services for automation.

The proposed data flow is as follows:

1.  **Ingestion:** A user or automated script uploads a PDF to the GCS data bucket.
2.  **Trigger:** The GCS upload event (specifically `google.storage.object.finalize`) triggers a Cloud Function.
3.  **Initiation:** The Cloud Function sends a message containing the PDF's GCS path to a `pdf-extraction-requests` Pub/Sub topic and initiates a Cloud Workflow execution.
4.  **Extraction Job:** The Cloud Workflow applies a Kubernetes Job manifest to the GKE cluster. This Job creates a pod that subscribes to the `pdf-extraction-requests` topic. The pod, running the Nougat container on a GPU node, processes the PDF and uploads the resulting `.mmd` file to a different GCS prefix.
5.  **Cleaning Job:** Upon successful completion, the extraction pod sends a message with the `.mmd` file's path to a `text-cleaning-requests` Pub/Sub topic.
6.  **Chunking & Saving:** The Cloud Workflow, having waited for the extraction step, then applies a second Kubernetes Job manifest. This Job creates a pod that pulls the message from the cleaning topic, downloads the `.mmd` file, performs all cleaning and chunking logic, and saves the final `_cleaned.jsonl` and `_garbage.jsonl` files to the processed data bucket on GCS.

# **Section 5 - Software Domain Design**

## **5.1 Software Application Domain Chart**

An architectural diagram would illustrate the flow between the GCP services described in Section 3.

## **5.2 Software Application Domain**

The pipeline is composed of two primary domains: Data Ingestion/Orchestration and Data Processing.

### **5.2.1 Domain: Data Ingestion & Orchestration**

This domain is responsible for receiving raw files and managing the overall pipeline workflow. The components in this domain represent the proposed software layer for automation and are not yet defined in the current `terraform/main.tf`.

- **Component: Google Cloud Functions:** Provides the event-driven trigger that initiates the pipeline upon file upload.
- **Component: Google Cloud Pub/Sub:** Acts as a message bus to decouple the pipeline stages, providing resilience and asynchronous processing.
- **Component: Google Cloud Workflows:** Orchestrates the sequence of jobs, state management, and error handling for the entire pipeline.

### **5.2.2 Domain: Data Processing**

This domain is responsible for the intensive computational tasks of OCR and data cleaning. The core component is provisioned by the `terraform/main.tf` file.

- **Component: Google Kubernetes Engine (GKE) Cluster:** Provides the scalable, containerized environment for running processing jobs. The cluster utilizes a specialized node pool with preemptible VMs and GPUs for cost-effective performance.
- **Task: Nougat OCR Job:** A containerized application that performs the initial PDF-to-Markdown conversion.
- **Task: Text Cleaning & Chunking Job:** A containerized application that performs the final data cleaning, filtering, and structuring.

# **Section 6 – Data Design**

## **6.1 Persistent/Static Data**

- **Raw PDFs:** The source material stored in a dedicated GCS bucket. This data is considered the immutable source of truth.
- **Processed `.mmd` files:** Intermediate Markdown files generated by Nougat, stored in a GCS prefix.
- **Final `.jsonl` files:** The primary output of the pipeline. These are newline-delimited JSON files stored in GCS, containing the cleaned and chunked text ready for LLM ingestion.
- **Garbage `.jsonl` files:** A secondary output containing text segments that were filtered out, stored for manual review and analysis.

## **6.2 Transient/Dynamic Data**

- **Pub/Sub Messages:** JSON-formatted messages that are passed between pipeline stages. Each message is small and contains metadata, such as the GCS path to the file that needs to be processed. This data is ephemeral.

## **6.3 External Interface Data**

This system does not have external data interfaces beyond the initial upload of PDFs to the GCS bucket.

## **6.4 Transformation of Data**

1.  **PDF to MMD:** The core transformation performed by the Nougat OCR tool, converting a binary PDF into structured Markdown.
2.  **MMD to JSONL:** The cleaning and chunking job transforms the raw Markdown into structured, cleaned, and chunked JSONL format. This involves heavy filtering, sanitization, and structural manipulation.

# **Section 7 - User Interface Design**

This is a backend data processing pipeline and has no user interface.

# **Section 8 - Other Interfaces**

The proposed automated system is composed of internal interfaces between GCP services. The interfaces involving Cloud Functions and Pub/Sub are part of the target architecture and are not yet defined in the current Terraform configuration.

- **GCS -> Cloud Functions:** An event-based interface where a file creation event in GCS triggers a Cloud Function.
- **Cloud Functions -> Pub/Sub:** A programmatic interface where the function publishes a message to a topic.
- **Pub/Sub -> GKE:** GKE pods subscribe to Pub/Sub topics to pull messages and initiate processing jobs.
- **GKE -> GCS:** GKE pods write their output (intermediate `.mmd` and final `.jsonl` files) back to GCS.

# **Section 9 - Extra Design Features / Outstanding Issues**

## **9.1 Security**

- **IAM Service Accounts:** GKE node pools are assigned a dedicated IAM Service Account with the principle of least privilege, granting necessary permissions to read from and write to GCS buckets.
- **GCS Bucket Security:** GCS buckets are private, with Uniform Bucket-Level Access enabled. Access is controlled exclusively through IAM.
- **Secret Manager:** Future secrets (e.g., API keys) will be stored in Google Secret Manager and accessed by GKE pods using Workload Identity.

## **9.2 Error Handling and Monitoring**

- **Dead-Letter Topics:** Each Pub/Sub subscription is configured with a dead-letter topic. If a message cannot be processed successfully, it is moved to the dead-letter topic for manual inspection.
- **Cloud's Operations Suite (formerly Stackdriver):**
    - **Cloud Logging:** All components write structured JSON logs to Cloud Logging for centralized analysis.
    - **Cloud Monitoring:** Key metrics (Pub/Sub queue depth, GKE node utilization, Job failure rates) are tracked in dashboards.
    - **Cloud Monitoring Alerts:** Alerts are configured to notify administrators of issues, such as a rise in the dead-letter topic size or a high rate of failed Kubernetes Jobs.

## **9.3 Future Improvements / Outstanding Issues**

- **Workflow Orchestration:** Fully implement the Cloud Workflows definition to replace any manual steps in the process.
- **Autoscaling:** Configure the GKE node pool to autoscale based on the number of pending jobs (messages in the Pub/Sub topics).
- **CI/CD Pipeline:** Implement a full CI/CD pipeline using Cloud Build to automatically build and deploy new container images to Artifact Registry upon commits to the main branch.

# **Section 10 – References**

- Project Infrastructure: `terraform/main.tf`
- Nougat OCR: [https://github.com/facebookresearch/nougat](https://github.com/facebookresearch/nougat)

# **Section 11 – Glossary**

- **GCP:** Google Cloud Platform
- **GCS:** Google Cloud Storage. The object storage service used for PDFs and processed data.
- **GKE:** Google Kubernetes Engine. The service used to orchestrate containerized processing jobs.
- **IaC:** Infrastructure as Code. Managing infrastructure through definition files (e.g., Terraform).
- **JSONL:** Newline-delimited JSON. A text format where each line is a valid JSON object.
- **Nougat:** The OCR model used for PDF text extraction.
- **OCR:** Optical Character Recognition.
- **Pub/Sub:** Google Cloud's asynchronous messaging service.
