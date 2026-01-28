#!/bin/bash

# ROS2 환경 설정
source /opt/ros/humble/setup.bash

echo "-----------------------------------------------------"
echo "ROS2 Perception Container is running."
echo "Models are located in /models"
echo "Your workspace is in /root/colcon_ws"
echo "-----------------------------------------------------"
echo ""

# 컨테이너가 종료되지 않도록 유지하면서 bash 셸 실행
exec /bin/bash