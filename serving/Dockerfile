FROM us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20240721_0916_RC00

# Argument to receive the Hugging Face token


# Health check and prediction ports for Vertex AI
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/v1/completions
ENV AIP_HTTP_PORT=8080

# Start the vLLM OpenAI-compatible server
CMD [ "python3", "-m", "vllm.entrypoints.openai.api_server",       "--host", "0.0.0.0",       "--port", "8080",       "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",       "--tensor-parallel-size", "2",       "--gpu-memory-utilization", "0.9" ]
