FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

# Add NVIDIA GPG keys (fixes CUDA repository issues)
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub || true
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub || true

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

WORKDIR /root/workspace
COPY ./workspace /root/workspace
RUN git clone https://github.com/dougsm/ggcnn.git
WORKDIR /root/workspace/ggcnn
RUN pip install -r requirements.txt

RUN pip install websockets

WORKDIR /root/workspace

