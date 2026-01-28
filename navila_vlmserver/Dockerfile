# Start from NVIDIA PyTorch base image with CUDA 12.1 and Python 3.10
# This provides Python 3.10+ for numpy 1.26.0 compatibility while using compatible CUDA version
FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set a non-interactive frontend for package installers
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory inside the container
WORKDIR /workspace

# Install system dependencies needed for git and building Python packages
# Also install Rust compiler needed for tokenizers build
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for tokenizers compilation)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Upgrade pip first to avoid compatibility issues
RUN pip install --upgrade pip setuptools wheel

# Install transformers as it's required but not in the base image
RUN pip install --no-cache-dir transformers

# --- NaVILA Repository Installation ---
# 1. Clone the specified repository
RUN git clone https://github.com/AnjieCheng/NaVILA.git

# 2. Set the working directory to the cloned repository
WORKDIR /workspace/NaVILA

# 3. Install the dependencies as specified in the README
# We chain these commands into a single RUN layer for efficiency.
# The user requested to skip the transformers re-installation.
# The `.[train]` extra should install deepspeed, which is required for the file copy later.
RUN pip install --no-cache-dir flash-attn==2.5.8 && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -e ".[train]" && \
    pip install --no-cache-dir -e ".[eval]"

# 4. Patch the installed transformers and deepspeed libraries
# This step is crucial for the model to work correctly as per the repo's instructions.
# We find the site-packages path and then copy the replacement files into it.
RUN site_pkg_path=$(python3 -c 'import site; print(site.getsitepackages()[0])') && \
    echo "Site packages found at: $site_pkg_path" && \
    cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/ && \
    cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

# --- Server Script and Model Setup ---
# 5. Go back to the main app directory
WORKDIR /workspace

# # 6. Copy the VLM server script into the working directory
# COPY vlm_server.py .

# 7. Copy the model checkpoints from the host's ./workspace directory
# to the container's /app/workspace directory
# COPY ./workspace/vlm_server.py /app/vlm_server.py

# Expose the port the server will listen on
EXPOSE 54321

# Define the command to run the application
# Use "0.0.0.0" as the host to make it accessible from outside the container.
# ENTRYPOINT ["python", "workspace/vlm_server.py"]
# CMD ["--model_path", "/app/workspace", "--host", "0.0.0.0", "--port", "54321"]