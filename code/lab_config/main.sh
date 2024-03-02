#!/bin/bash
source /opt/ros/melodic/setup.bash
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
python3 /home/ubuntu/lab_config/main.py
