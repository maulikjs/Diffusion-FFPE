# Use CUDA development image instead of runtime
FROM nvidia/cuda:12.6.0-devel-ubuntu20.04

# Set working directory and environment variables
WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

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

# Remove any existing CuPy installations and install the correct version
RUN pip uninstall -y cupy-cuda11x cupy-cuda12x || true && \
    pip uninstall -y cupy || true && \
    pip install --no-cache-dir cupy-cuda12x==12.3.0

# Copy requirements first for better layer caching
COPY deployment/requirements.txt .

# Install requirements with --ignore-installed flag to avoid uninstall errors
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# Install PyTorch with CUDA 12.6 support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install xformers with CUDA 12.6 support
RUN pip install xformers

# Create directories for data and outputs
RUN mkdir -p /data/uploads /data/results /app/checkpoints

# Copy application code
COPY deployment/__init__.py /app/
COPY inference_ddp.py /app/
COPY diffusion_ffpe /app/diffusion_ffpe/
COPY vision_aided_loss /app/vision_aided_loss
COPY open_clip_custom /app/open_clip_custom
COPY checkpoints /
COPY deployment/processing_controller.py /app/deployment/
COPY deployment/model_definitions.py /app/deployment/
COPY deployment/api_server.py /app/deployment/
COPY deployment/chunk_extraction.py /app/deployment
COPY deployment/stitch_image.py /app/deployment
COPY deployment/setup_logger.py /app/deployment


# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CHECKPOINT_DIR=/app/checkpoints
ENV UPLOAD_DIR=/data/uploads
ENV RESULTS_DIR=/data/results
ENV CUDA_VISIBLE_DEVICES=0,1

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