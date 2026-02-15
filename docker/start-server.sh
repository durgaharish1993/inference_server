#!/bin/bash
set -e

# Start Triton server in background
echo "Starting Triton Inference Server..."
tritonserver --model-repository=/workspace/triton_serve --log-verbose=1 &

# Wait for Triton to be ready
echo "Waiting for Triton server to be ready..."
until curl -f http://localhost:8000/v2/health/ready; do
    echo "Waiting for Triton server..."
    sleep 5
done

echo "Triton server is ready!"

# Start FastAPI server
echo "Starting FastAPI server..."
cd /workspace/server
python3 main.py