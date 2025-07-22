# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python, pip, and git
RUN apt-get update && apt-get install -y --no-install-recommends     python3.10 python3-pip git &&     rm -rf /var/lib/apt/lists/*

# Install vLLM and other dependencies
RUN pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install vllm==0.2.6 huggingface_hub

# Argument to receive the Hugging Face token
ARG HF_TOKEN
ENV HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Health check and prediction ports for Vertex AI
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/v1/completions
ENV AIP_HTTP_PORT=8080

# Start the vLLM OpenAI-compatible server
# It will download the model on first startup.
# TENSOR_PARALLEL_SIZE is the key to splitting the model across GPUs.
CMD [ "python3", "-m", "vllm.entrypoints.openai.api_server",       "--host", "0.0.0.0",       "--port", "8080",       "--model", "meta-llama/Llama-2-13b-chat-hf",       "--tensor-parallel-size", "2" ]
