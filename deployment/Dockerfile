# Dockerfile for WSI Processing Pipeline with CUDA 12.6 support
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install Python and required system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openslide-tools \
    libopenjp2-7 \
    libwebp-dev \
    python3-openslide \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python3 as default
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better layer caching
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.6 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Create directories for data and outputs
RUN mkdir -p /data/uploads /data/results /app/checkpoints

# Copy application code
# First the main scripts
COPY chunk_extraction.py /app/
COPY stitch_image.py /app/
COPY inference_ddp.py /app/

# Then the deployment files
COPY deployment/processing_controller.py /app/deployment/
COPY deployment/model_definitions.py /app/deployment/
COPY deployment/api_server.py /app/deployment/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CHECKPOINT_DIR=/app/checkpoints
ENV UPLOAD_DIR=/data/uploads
ENV RESULTS_DIR=/data/results
ENV CUDA_VISIBLE_DEVICES=0,1,2,3

# Test GPU access
RUN echo "Testing NVIDIA GPUs:" && \
    nvidia-smi || echo "Note: nvidia-smi may fail in build, but should work when container runs with proper device mappings"

# Define mount points for volumes
VOLUME ["/app/checkpoints"]
VOLUME ["/data/uploads", "/data/results"]

# Expose port for API
EXPOSE 5000

# Start the API server
CMD ["python", "deployment/api_server.py"]