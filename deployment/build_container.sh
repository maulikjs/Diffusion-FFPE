#!/bin/bash
# Script to build the docker container

# Get the absolute path to the project root
PROJECT_ROOT=$(realpath $(dirname "$0")/..)
echo "Project root: $PROJECT_ROOT"

# Navigate to project root
cd $PROJECT_ROOT

# Build the docker image
echo "Building Docker image from project root..."
docker build -t wsi-processor:latest -f deployment/Dockerfile .

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "You can start the container with the GPU device mappings using:"
    echo "cd deployment && ./start-container.sh"
else
    echo "Build failed!"
fi