#!/bin/bash
# Script to start the container with explicit GPU device mappings

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Please make sure NVIDIA drivers are installed."
    exit 1
fi

# Get the number of available GPUs
# GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
# echo "Detected $GPU_COUNT GPU(s)"

#TODO revert to previous
GPU_COUNT=1

if [ $GPU_COUNT -eq 0 ]; then
    echo "Error: No NVIDIA GPUs detected."
    exit 1
fi

# Base docker run command
DOCKER_CMD="docker run -d \
    --name wsi-processor \
    -p 5000:5000 \
    --shm-size=16gb \
    -v $(pwd)/../data/uploads:/data/uploads \
    -v $(pwd)/../data/results:/data/results \
    -v $(pwd)/../checkpoints:/app/checkpoints"

# Add environment variables
DOCKER_CMD="$DOCKER_CMD \
    -e CHECKPOINT_DIR=/app/checkpoints \
    -e UPLOAD_DIR=/data/uploads \
    -e RESULTS_DIR=/data/results \
    -e MAX_WORKERS=2 \
    -e GPU_COUNT=$GPU_COUNT"

# # Add all available GPUs
# for ((i=0; i<$GPU_COUNT; i++)); do
#     DOCKER_CMD="$DOCKER_CMD \
#     --device /dev/nvidia$i:/dev/nvidia$i"
# done

#TODO to revert to previous
DOCKER_CMD="$DOCKER_CMD \
    --device /dev/nvidia1:/dev/nvidia1"

# Add NVIDIA control devices
DOCKER_CMD="$DOCKER_CMD \
    --device /dev/nvidiactl:/dev/nvidiactl \
    --device /dev/nvidia-uvm:/dev/nvidia-uvm"

# Add image name
DOCKER_CMD="$DOCKER_CMD wsi-processor:latest"

# Execute the command
echo "Starting container with the following command:"
echo "$DOCKER_CMD"
eval $DOCKER_CMD

echo "Container started. Check status with: docker ps"
echo "View logs with: docker logs wsi-processor"