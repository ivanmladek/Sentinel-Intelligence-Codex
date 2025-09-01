#!/bin/bash

# Script to build and run the Docker container locally for debugging

# Build the Docker image
echo "Building Docker image..."
docker build -t pdf-processor:local -f docker/Dockerfile.processing .

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Docker build failed!"
    exit 1
fi

# Run the container with sample environment variables
# You can modify these values as needed for your testing
echo "Running container..."
docker run --rm \
  -e BASE_URL="https://the-eye.eu/public/Books/Bibliotheca%20Alexandrina/" \
  -e BUCKET_NAME="your-test-bucket-name" \
  -e NUM_PODS=1 \
  -e JOB_COMPLETION_INDEX=0 \
  pdf-processor:local

# For interactive debugging, you can use:
# docker run -it --rm \
#   -e BASE_URL="https://the-eye.eu/public/Books/Bibliotheca%20Alexandrina/" \
#   -e BUCKET_NAME="your-test-bucket-name" \
#   -e NUM_PODS=1 \
#   -e JOB_COMPLETION_INDEX=0 \
#   pdf-processor:local /bin/bash