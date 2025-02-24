#!/bin/bash
# Script to stop and clean up the WSI processor container

# Stop the running container
echo "Stopping wsi-processor container..."
docker stop wsi-processor

# Remove the container
echo "Removing wsi-processor container..."
docker rm wsi-processor

echo "Container stopped and removed."
echo "You can start it again with: ./start-container.sh"