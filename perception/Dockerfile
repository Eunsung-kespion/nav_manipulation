# Base image: ROS2 Humble Desktop Full
FROM osrf/ros:humble-desktop-full

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
# Environment variables for GroundingDINO build
ENV AM_I_IN_A_DOCKER_CONTAINER=1
ENV TORCH_HOME=/root/.cache/torch

# Install system dependencies
RUN apt-get update && apt-get install -y \
    net-tools \
    python3-pip \
    python3-dev \
    python-is-python3 \
    git \
    wget \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    nvidia-cuda-toolkit \
    gcc-9 \
    g++-9 \
    && rm -rf /var/lib/apt/lists/*

# Apply ROS environment to all RUN, CMD, ENTRYPOINT commands
SHELL ["/bin/bash", "-c"]

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# CUDA toolkit installed via APT is located at /usr
ENV CUDA_HOME=/usr
ENV PATH=/usr/bin:$PATH

# Install Python dependencies (PyTorch + Track-Anything + Grounded-SAM)
# Install PyTorch and torchvision first
RUN source /opt/ros/humble/setup.bash && pip3 install --no-cache-dir \
    torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Install remaining Python packages (limit numpy<2 to maintain compatibility with MinkowskiEngine)
RUN source /opt/ros/humble/setup.bash && pip3 install --no-cache-dir \
    "numpy<2" \
    opencv-python-headless \
    scipy \
    matplotlib \
    pyyaml \
    timm \
    einops \
    scikit-image \
    onnx \
    onnxruntime-gpu \
    trimesh \
    supervision

# Git clone Perception Module models
WORKDIR /models

RUN source /opt/ros/humble/setup.bash && apt-get update
RUN source /opt/ros/humble/setup.bash && apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN source /opt/ros/humble/setup.bash && apt-get clean
RUN source /opt/ros/humble/setup.bash && rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
ENV MAX_JOBS=12
RUN source /opt/ros/humble/setup.bash && git clone --recursive "https://github.com/leadbreak/MinkowskiEngine"
# Use gcc-9/g++-9 for MinkowskiEngine build (for compatibility with CUDA toolkit)
RUN source /opt/ros/humble/setup.bash && \
    cd MinkowskiEngine && \
    CC=gcc-9 CXX=g++-9 python setup.py install --force_cuda --blas=openblas

WORKDIR /models
COPY ./workspace/anygrasp_sdk /models/anygrasp_sdk

WORKDIR /models/anygrasp_sdk/
# Allow sklearn deprecated packages (resolves graspnetAPI dependency)
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN source /opt/ros/humble/setup.bash && pip install -r requirements.txt

WORKDIR /models/anygrasp_sdk/pointnet2  
RUN source /opt/ros/humble/setup.bash && \
    CC=gcc-9 CXX=g++-9 python setup.py install

WORKDIR /models/anygrasp_sdk/grasp_detection
RUN cp gsnet_versions/gsnet.cpython-310-x86_64-linux-gnu.so gsnet.so
RUN cp ../license_registration/lib_cxx_versions/lib_cxx.cpython-310-x86_64-linux-gnu.so lib_cxx.so

WORKDIR /models/anygrasp_sdk/grasp_tracking
RUN cp tracker_versions/tracker.cpython-310-x86_64-linux-gnu.so tracker.so
RUN cp ../license_registration/lib_cxx_versions/lib_cxx.cpython-310-x86_64-linux-gnu.so lib_cxx.so

# Add OpenSSL 1.1 runtime (for libcrypto.so.1.1) on Ubuntu 22.04
RUN set -eux; \
    apt-get update; apt-get install -y --no-install-recommends ca-certificates wget; \
    echo "deb [arch=amd64] http://security.ubuntu.com/ubuntu focal-security main" \
      > /etc/apt/sources.list.d/focal-security.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends libssl1.1; \
    rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/focal-security.list

# TODO: Place ROS driver in colcon_ws
# Set final working directory
WORKDIR /root/colcon_ws
RUN mkdir -p src