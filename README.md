# Ackermann Autonomous Car Simulation in gazebo 

## → Requirements

Before building and running the package, install the necessary dependencies using the `$ROS_DISTRO` environment variable:

```python
sudo apt-get install libeigen3-dev
sudo apt install ros-$ROS_DISTRO-joint-state-publisher-gui
sudo apt install ros-$ROS_DISTRO-xacro
sudo apt install ros-$ROS_DISTRO-gazebo-ros-pkgs
sudo apt install ros-$ROS_DISTRO-ros2-control ros-$ROS_DISTRO-ros2-controllers
sudo apt install ros-$ROS_DISTRO-controller-manager
```

### → Step 1: Create a ROS2 Workspace

1. Open a terminal and **Set up ROS2**.
    
    ```bash
    source /opt/ros/foxy/setup.bash #for ros2 foxy
    source /opt/ros/humble/setup.bash #for ro2 humble
    ```
    
2. Create a directory for the workspace and navigate into it:
    
    ```bash
    mkdir -p ~/ros2_ws/src
    cd ~/ros2_ws
    ```
# ROS2 Lane Detection and STOP Sign Detection Nodes

## Description

This repository contains two ROS2 nodes designed for autonomous driving tasks: 

1. **Lane Detection Node**:  
   This node performs lane detection using a YOLOv8 segmentation model. It processes camera input to identify lanes and uses a PID controller to generate control commands for maintaining the vehicle's trajectory.

2. **STOP Sign Detection Node**:  
   This node detects STOP signs using a YOLOv8 object detection model. It processes camera input, identifies STOP signs in the environment, and provides real-time feedback about detection.

## Requirements

To run these nodes, ensure the following prerequisites are installed:

- **ROS2 Humble or later**  
- **Python 3.8+**  
- Required Python libraries:
  - `rclpy`
  - `sensor_msgs`
  - `geometry_msgs`
  - `cv_bridge`
  - `opencv-python`
  - `torch`
  - `ultralytics`
  - `numpy`
- A compatible camera publishing images to a ROS2 topic (e.g., `/camera/camera/color/image_raw`).

## YOLOv8 Models

Both nodes require pre-trained YOLOv8 models:
- Lane Detection Node: Model file named `best.pt`.
- STOP Sign Detection Node: Model file named `best_stop.pt`.

Place the models in a directory named `models` in the root of the project.

## How to Run

### Lane Detection Node

1. Ensure the YOLOv8 lane detection model is available at `models/best.pt`.  
2. Run the node:
   ```bash
   ros2 run <package_name> lane_detection_node

### STOP Sign Detection Node

1. Ensure the YOLOv8 STOP sign detection model is available at models/best_stop.pt
2. Run the node:
   ```bash
   ros2 run <package_name> stop_sign_node
