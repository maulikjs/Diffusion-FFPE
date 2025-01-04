FROM python:3.9-slim

# Install system dependencies for OpenSlide, etc.
RUN apt-get update && apt-get install -y \
    openslide-tools \
    libopenslide0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /tmp
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy your chunking script
COPY setup_logger.py /app/setup_logger.py
COPY chunking.py /app/chunking.py
WORKDIR /app

# We'll use Python as the entrypoint
ENTRYPOINT ["python", "/app/chunking.py"]
