FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV NVIDIA_VISIBLE_DEVICES=all
# Environment variables for GroundingDINO build
ENV AM_I_IN_A_DOCKER_CONTAINER=1
ENV TORCH_HOME=/root/.cache/torch

RUN apt-get update -y || true && \
    apt-get install -y --no-install-recommends \
    wget \
    gnupg

RUN apt-get update -y || true && \
    apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get clean

RUN rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -f /etc/apt/sources.list.d/nvidia-ml.list


# Keys are now registered, so update and install will succeed
RUN apt-get update && apt-get install -y \
    net-tools \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Apply ROS environment to all RUN, CMD, ENTRYPOINT commands
SHELL ["/bin/bash", "-c"]

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# CUDA toolkit installed via APT is located at /usr
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

# Install remaining Python packages (limit numpy<2 to maintain compatibility with MinkowskiEngine)
RUN pip3 install --no-cache-dir \
    "numpy==1.23.4" \
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

RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# For faster build, use more jobs.
# ENV MAX_JOBS=12
RUN git clone https://github.com/ChenN-Scott/MotionGrasp.git

COPY ./replaces/requirements.txt /models/MotionGrasp/requirements.txt
RUN cd MotionGrasp && pip install -r requirements.txt

WORKDIR /models/MotionGrasp/knn
RUN /opt/conda/bin/python setup.py install

WORKDIR /models/MotionGrasp/pointnet2
RUN /opt/conda/bin/python setup.py install

RUN mkdir -p /models/MotionGrasp/log/log_rs
COPY ./workspace/checkpoint-rs.tar /models/MotionGrasp/log/log_rs/checkpoint-rs.tar

# TODO: Place ROS driver in colcon_ws
# Set final working directory