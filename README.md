# Navigation & Manipulation Docker Services

This repository contains Docker services for robotics perception, grasping, and vision-language model inference. Each subdirectory provides a containerized environment for different perception and manipulation tasks.

> **Note:** This repository was used for personal projects and may not receive active maintenance or support. Use at your own discretion.

## Prerequisites

- Docker Engine (20.10+) & Docker Compose (1.29+)
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit
- X11 server for GUI applications

```bash
# Enable X11 forwarding
xhost +local:docker
# For Wayland users:
xhost +si:localuser:root
```

## Services Overview

### 1. Perception Dataset
**Directory:** `perception_dataset/`

MATLAB R2020b environment for processing and visualizing multi-object grasp datasets with RGBD data.

**Base Image:** `mathworks/matlab:r2020b`

---

### 2. Perception GGCNN
**Directory:** `perception_ggcnn/`

Generative Grasping Convolutional Neural Network for robotic grasp detection with pre-trained Cornell weights.

**Base Image:** `pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel`

**Features:**
- WebSocket support for real-time inference
- Pre-trained GGCNN/GGCNN2 models

---

### 3. Perception VGN
**Directory:** `perception_vgn/`

Volumetric Grasping Network for 3D grasp pose estimation from point clouds with Panda robot integration.

**Base Image:** `nvidia/cuda:12.0.1-devel-ubuntu20.04`

**Key Scripts:** `generate_data.py`, `train_vgn.py`, `sim_grasp.py`, `panda_grasp.py`

---

### 4. Perception FastSAM
**Directory:** `perception_fastsam/`

Fast Segment Anything Model for real-time segmentation with WebSocket server support.

**Base Image:** `pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel`

**Ports:** 8766 (WebSocket)

**Note:** Place model weights (`FastSAM.pt`) in `./weights/` directory.

---

### 5. GORM TF
**Directory:** `gorm_tf/`

ROS2 Humble-based service for geometric object relationship mapping and transformation.

**Base Image:** `osrf/ros:humble-desktop-full`

---

### 6. Perception MotionGrasp
**Directory:** `perception_motiongrasp/`

MotionGrasp for dynamic grasp prediction with motion information using PointNet2 backend.

**Base Image:** `pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel`

---

### 7. Perception (AnyGrasp)
**Directory:** `perception/`

AnyGrasp SDK for robust grasp detection and tracking with MinkowskiEngine backend.

**Base Image:** `osrf/ros:humble-desktop-full`

**License Required:** Commercial license (user: EunsungKim)

**Components:**
- Grasp detection with GSNet
- Grasp tracking with temporal consistency

---

### 8. NaVILA VLM Server
**Directory:** `navila_vlmserver/`

NaVILA Vision-Language Model server for navigation and instruction following with LLaMA3-8B.

**Base Image:** `nvcr.io/nvidia/pytorch:24.01-py3`

**Ports:** 54321 (VLM Server)

**Features:**
- Flash Attention 2.5.8
- Vision-language understanding
- Pre-trained NaVILA-LLaMA3-8B checkpoint

**Model Setup:**

Download the pre-trained model from Hugging Face:

```bash
cd navila_vlmserver/workspace
git clone https://huggingface.co/a8cheng/navila-llama3-8b-8f
```

Or using Hugging Face CLI:

```bash
cd navila_vlmserver/workspace
huggingface-cli download a8cheng/navila-llama3-8b-8f --local-dir navila-llama3-8b-8f
```

Model source: [https://huggingface.co/a8cheng/navila-llama3-8b-8f](https://huggingface.co/a8cheng/navila-llama3-8b-8f)

---

## Common Configuration

All services share:
- **GPU Access:** Full NVIDIA GPU access
- **Network Mode:** Host networking for ROS/WebSocket
- **Display:** X11 forwarding for GUI applications
- **Hardware Access:** Direct `/dev` access for cameras
- **IPC/PID:** Host namespaces for efficient communication

## Quick Start

```bash
# Navigate to service directory
cd <service_directory>

# Start the service
docker-compose up -d

# Enter the container
docker exec -it <container_name> bash

# Stop the service
docker-compose down

# Rebuild after changes
docker-compose build --no-cache && docker-compose up -d
```

## Hardware Requirements

**Minimum:**
- NVIDIA GPU with CUDA Compute Capability 7.0+
- VRAM: 8GB+ (16GB for VLM models)
- RAM: 16GB+ (32GB recommended)
- Storage: 100GB+ for all services

**Tested on:**
- NVIDIA RTX 3090/4090, RTX A6000
- CUDA Compute Capability 8.0, 8.6

**Camera Support:**
- Intel RealSense D435/D455
- Generic RGB-D cameras

## Troubleshooting

**GPU Issues:**
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

**X11 Display Issues:**
```bash
xhost +local:docker
export DISPLAY=:0
```

**ROS Communication Issues:**
Ensure `network_mode: host` is set in docker-compose.yml.

## License

Each service has its own license:
- **AnyGrasp SDK:** Commercial license required
- **GGCNN:** Check original repository
- **VGN:** MIT License
- **FastSAM:** Apache 2.0 License
- **NaVILA:** Academic use

## References

- [GGCNN Repository](https://github.com/dougsm/ggcnn)
- [VGN Repository](https://github.com/ethz-asl/vgn)
- [FastSAM Repository](https://github.com/CASIA-IVA-Lab/FastSAM)
- [MotionGrasp Repository](https://github.com/ChenN-Scott/MotionGrasp)
- [NaVILA Repository](https://github.com/AnjieCheng/NaVILA)
- [MinkowskiEngine Repository](https://github.com/NVIDIA/MinkowskiEngine)

---

**Disclaimer:** This repository was created for personal research projects and may not receive ongoing maintenance or support. Please refer to the original repositories for official implementations and updates.
