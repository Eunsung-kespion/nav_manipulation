#!/bin/bash
set -e

# ROS 2 기본 환경 설정
source /opt/ros/humble/setup.bash

# 생성한 워크스페이스 환경 설정
source /ros2_ws/install/setup.bash

# Docker CMD로 전달된 명령 실행
exec "$@"