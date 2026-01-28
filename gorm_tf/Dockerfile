FROM osrf/ros:humble-desktop-full

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Install system and Python dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-joint-state-publisher-gui \
    ros-humble-rqt-tf-tree \
    nano \
    && rm -rf /var/lib/apt/lists/*

RUN pip install scipy numpy

# Create ROS 2 workspace
ARG ROS_WS=/ros2_ws
RUN mkdir -p ${ROS_WS}/src

# Set working directory
WORKDIR ${ROS_WS}

# Install ROS dependencies (based on package.xml)
RUN source /opt/ros/humble/setup.bash 

# Build workspace
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# Copy entrypoint script and grant execute permission
COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /ros2_ws/install/setup.bash" >> ~/.bashrc

CMD ["bash"]