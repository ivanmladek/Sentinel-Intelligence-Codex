FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip

# Install PyTorch (CUDA 12.x compatible)
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
RUN pip3 install vllm

# Health check and prediction ports for Vertex AI
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/v1/completions
ENV AIP_HTTP_PORT=8080

# Start the vLLM OpenAI-compatible server
CMD [ "python3", "-m", "vllm.entrypoints.openai.api_server",       "--host", "0.0.0.0",       "--port", "8080",       "--model", "meta-llama/Llama-2-13b-chat-hf",       "--tensor-parallel-size", "2",       "--gpu-memory-utilization", "0.9" ]
