# Ackermann Autonomous Car Simulation 

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
    