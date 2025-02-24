#!/bin/bash
# Script to test the WSI Processing API without a UI

# Configuration
API_URL="http://localhost:5000/api"
SVS_FILE="/home/maulik/work/Diffusion-FFPE/deployment/tmp/test-small.svs"  # Replace with path to your WSI file

# Color output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}WSI Processing API Test Script${NC}"
echo "======================================="

# 1. Check API health
echo -e "\n${YELLOW}Checking API health...${NC}"
HEALTH_RESPONSE=$(curl -s ${API_URL}/health)
if [[ $HEALTH_RESPONSE == *"status":"ok"* ]]; then
  echo -e "${GREEN}API is healthy!${NC}"
else
  echo -e "${RED}API health check failed:${NC}"
  echo $HEALTH_RESPONSE
fi

# 2. List available models
echo -e "\n${YELLOW}Listing available models...${NC}"
MODELS_RESPONSE=$(curl -s ${API_URL}/models)
echo $MODELS_RESPONSE | python -m json.tool

# 3. Upload file and start processing
echo -e "\n${YELLOW}Uploading file for processing...${NC}"
UPLOAD_RESPONSE=$(curl -s -X POST \
  -F "file=@${SVS_FILE}" \
  -F "chunk_size=512" \
  -F "num_workers=2" \
  ${API_URL}/upload)

# Extract job_id from response
JOB_ID=$(echo $UPLOAD_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('job_id', ''))")

if [ -z "$JOB_ID" ]; then
  echo -e "${RED}Failed to get job ID from response:${NC}"
  echo $UPLOAD_RESPONSE
  exit 1
fi

echo -e "${GREEN}Processing started with job ID: ${JOB_ID}${NC}"

# 4. Poll for status
echo -e "\n${YELLOW}Polling for job status...${NC}"
STATUS="running"
PROGRESS=0

while [[ "$STATUS" == "running" || "$STATUS" == "idle" ]]; do
  STATUS_RESPONSE=$(curl -s ${API_URL}/status/${JOB_ID})
  STATUS=$(echo $STATUS_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('status', ''))")
  NEW_PROGRESS=$(echo $STATUS_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('progress', 0))")
  MESSAGE=$(echo $STATUS_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin).get('message', ''))")
  
  # Only print if progress changed
  if (( $(echo "$NEW_PROGRESS > $PROGRESS" | bc -l) )); then
    PROGRESS=$NEW_PROGRESS
    echo -e "Progress: ${PROGRESS}% - ${MESSAGE}"
  fi
  
  sleep 5
done

# 5. Final status
if [[ "$STATUS" == "complete" ]]; then
  echo -e "\n${GREEN}Processing completed successfully!${NC}"
  echo -e "Download the result with: curl -o output.tiff ${API_URL}/result/${JOB_ID}"
  
  # Optional: automatically download the result
  echo -e "\n${YELLOW}Downloading result...${NC}"
  curl -o output.tiff ${API_URL}/result/${JOB_ID}
  echo -e "${GREEN}Result saved as output.tiff${NC}"
else
  echo -e "\n${RED}Processing failed with status: ${STATUS}${NC}"
  echo $STATUS_RESPONSE
fi

echo -e "\n${YELLOW}Test complete!${NC}"
