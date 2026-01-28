# FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

# Set working directory
WORKDIR /app

# Add NVIDIA GPG keys (fixes CUDA repository issues)
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub || true
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub || true

# Install system packages and git (including OpenCV dependencies)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA_HOME environment variable (default path for PyTorch devel image)
ENV CUDA_HOME=/usr/local/cuda

# Install Python libraries
# Install required libraries first: WebSocket, Pillow, opencv, etc.
RUN pip install --no-cache-dir websockets Pillow opencv-python numpy

# --- FastSAM Installation ---
# Create workspace directory for project code
RUN mkdir -p /app/workspace
WORKDIR /app/workspace

# Clone and install FastSAM repository
RUN git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
WORKDIR /app/workspace/FastSAM
RUN pip install -r requirements.txt
RUN pip install -e .

WORKDIR /app
RUN pip install git+https://github.com/openai/CLIP.git

# Move to main directory for FastAPI server code and pipeline script
WORKDIR /app

# SAM_pipelines.py: provided script
# main.py: FastAPI server to be written
COPY ./SAM_pipelines.py /app/
COPY ./main.py /app/

# Model checkpoints are large, so they should not be included in the Docker image.
# It is recommended to mount them via volume in docker-compose.yml.
# Example: ./weights:/app/weights (FastSAM.pt model file)
RUN mkdir -p /app/weights
RUN mkdir -p /app/outputs
RUN mkdir -p /app/temp_data

# Expose port
EXPOSE 8766