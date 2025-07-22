# Your Step-by-Step Guide to Launching a Multi-GPU AI Model on Google Cloud

Welcome! This guide will walk you through the entire process of deploying a powerful, multi-GPU AI model from your local machine onto Google Cloud Platform (GCP). Think of your computer as "mission control" and this guide as your launch sequence checklist.

### **Concept: What Are We Doing?**

We are going to run a single Python script (`multi_gpu_inference_orchestrator.py`) that acts as an instruction manual for Google Cloud. It will automatically command Google to:

1.  Set up a powerful server with **2 NVIDIA A100 GPUs**.
2.  Download and deploy a large, 13-billion parameter AI model (`Llama-2-13B`) onto that server, configured to run across both GPUs.
3.  Build a simple web application that lets you chat with your newly deployed AI model.
4.  Set up all the surrounding cloud infrastructure, including monitoring, networking, and batch processing capabilities.

---

### **Phase 1: Getting Your Local Machine Ready (Prerequisites)**

Before you can command Google, your own computer needs a few essential tools.

1.  **Google Cloud SDK (`gcloud`):** This is the main "remote control" for your GCP account from your terminal.
    *   **How to get it:** Follow the official instructions here: [Google Cloud SDK Installation](https://cloud.google.com/sdk/docs/install)

2.  **Docker Desktop:** We need to package our code into "containers" (like shipping containers for software). Docker is the standard tool for this.
    *   **How to get it:** Download and install it from the official site: [Docker Desktop](https://www.docker.com/products/docker-desktop/)

3.  **Hugging Face Account & Token:** The AI model we want to use is stored on a site called Hugging Face. We need an account and a special password (an "Access Token") to download it.
    *   **Create an account:** [huggingface.co](https://huggingface.co/join)
    *   **Get the token:** Go to your Hugging Face profile -> Settings -> Access Tokens -> New token. Give it a name and `read` permissions. Copy this token immediately and save it somewhere safe. You'll need it when you run the script.

---

### **Phase 2: Setting Up Your Google Cloud Project**

This is the most important phase. We need to give your GCP account the right permissions and enable the necessary services.

#### **Step 1: Log in and Select Your Project**

1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  At the top of the page, select the project you want to work in, or create a new one.
3.  **Crucially, make sure billing is enabled for this project.** You can check this by searching for "Billing" in the console. Without it, nothing will work.

#### **Step 2: Enable the Required APIs**

APIs are like on/off switches for different GCP services. We need to turn on all the ones our script uses.

1.  In the GCP Console, find the search bar at the top and search for **"APIs & Services"**.
2.  Click the **"+ ENABLE APIS AND SERVICES"** button at the top.
3.  Search for and **Enable** each of the following APIs, one by one:
    *   `Vertex AI API`
    *   `Artifact Registry API`
    *   `Cloud Run Admin API`
    *   `Kubernetes Engine API`
    *   `Cloud Build API`
    *   `Cloud Storage API`

#### **Step 3: Set Permissions (The "IAM" part)**

You need to give yourself permission to create all these resources. This is done in the **"IAM & Admin"** section.

1.  In the GCP Console search bar, type **"IAM"** and go to the IAM page.
2.  Find your user account in the list (it will be your email address). Click the **pencil icon** (Edit principal) next to your name.
3.  A panel will appear on the right. Click **"Add another role"**.
4.  Search for and add each of the following roles. This gives you the "keys" to run everything in the script:
    *   **Vertex AI Admin**
    *   **Storage Admin**
    *   **Cloud Build Editor**
    *   **Artifact Registry Admin**
    *   **Cloud Run Admin**
    *   **Kubernetes Engine Admin**
    *   **Service Account User**
5.  Click **"Save"**.

#### **Step 4: ❗ Request GPU Quota (CRITICAL STEP) ❗**

This is the step that stops most people. By default, Google doesn't let you use powerful GPUs to prevent accidental high bills. You have to ask for permission first.

1.  In the GCP Console search bar, type **"Quotas"** and go to the "IAM & Admin -> Quotas" page.
2.  In the "Filter" box at the top of the Quotas page, enter **`NVIDIA A100 40GB`**.
3.  You will see a list of quotas for that GPU type in different regions. Find the region you are using (the default in the script is `us-central1`).
4.  The "Limit" will probably be `0` or `1`. You need at least `2`.
5.  Select the checkbox next to the `us-central1` quota and click **"EDIT QUOTAS"** at the top.
6.  Fill out the form on the right. Set the "New limit" to `2`. In the description, explain your request: *"I am deploying a large language model on Vertex AI for a proof-of-concept and require 2 A100 40GB GPUs for a single machine."*
7.  Submit the request. **This can take a few hours or even a day or two for Google to approve.** You cannot proceed until this is approved.

---

### **Phase 3: Running the Orchestration Script**

Once your GCP project is set up and your GPU quota is approved, you're ready to run the script.

1.  **Open Your Terminal:** This is the command-line interface on your computer (e.g., Terminal on Mac/Linux, PowerShell or WSL on Windows).

2.  **Navigate to the Code:** Use the `cd` command to go into the project directory.
    ```bash
    cd /path/to/your/pirateGPT
    ```

3.  **Connect Your Terminal to GCP:** Run these two commands. A browser window will pop up for you to sign in with your Google account.
    ```bash
    gcloud auth login
    gcloud auth application-default login
    ```

4.  **Install Python Libraries:** Run this command to install the Python packages the script needs.
    ```bash
    pip install google-cloud-aiplatform "requests<3.0.0" flask gunicorn google-cloud-secret-manager
    ```

5.  **Configure The Script:** Open the `multi_gpu_inference_orchestrator.py` file in a text editor. **You must change the `PROJECT_ID` variable on line 8 to your actual GCP Project ID.**
    ```python
    # multi_gpu_inference_orchestrator.py

    # --- CONFIGURATION ---
    # PLEASE REPLACE WITH YOUR VALUES
    PROJECT_ID = "your-actual-gcp-project-id-goes-here" # <--- CHANGE THIS LINE
    # ...
    ```

6.  **Run the Script!** Now, execute the script from your terminal.
    ```bash
    python multi_gpu_inference_orchestrator.py
    ```
    *   It will first ask for your **Hugging Face Token**. Paste the token you saved earlier and press Enter.
    *   Now, be patient. The script will print updates as it commands GCP to build everything. This will take a long time (30-60 minutes is possible, especially the Docker build step).

---

### **Phase 4: Success and Cleanup**

If everything works, the script will finish by printing a URL and a `curl` command. You can run that `curl` command in your terminal to test your live AI model!

**💰💰 MOST IMPORTANT STEP: CLEAN UP!! 💰💰**

The resources you created are expensive. You must delete them to avoid a large bill. The script will print a list of resources to delete at the end. Here is how you find them in the GCP Console:

1.  **Delete the Vertex AI Endpoint:** Search for "Vertex AI", go to "Endpoints", find your endpoint (`llama2-13b-vllm-demo`), and delete it.
2.  **Delete the GKE Cluster:** Search for "Kubernetes Engine", find your cluster (`batch-inference-cluster`), and delete it.
3.  **Delete the Cloud Run Service:** Search for "Cloud Run", find your service (`langchain-llama2-frontend`), and delete it.
4.  **Delete the Container Images:** Search for "Artifact Registry", find the repository (`vllm-serving-repo`), and delete the images inside it.
5.  **Delete the GCS Bucket:** Search for "Cloud Storage", find your bucket (`your-project-id-vllm-13b-demo-bucket`), and delete it.
