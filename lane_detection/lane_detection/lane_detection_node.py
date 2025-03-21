import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from geometry_msgs.msg import Twist
import os
import numpy as np


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Load the YOLOv8 model with segmentation
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models', 'best.pt')

        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ YOLO model not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.get_logger().info(f"✅ Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path, task="segment")  # Force segmentation mode

        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Camera topic
            self.image_callback,
            10)
        
        # Publishers for the detected image and movement command
        self.publisher = self.create_publisher(Image, 'lane_detection/output', 10)
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # PID controller variables
        self.prev_error = 0.0
        self.integral = 0.0

    def image_callback(self, msg):
        # Convert the ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Display the received image
        if os.getenv("DISPLAY") is not None:
            cv2.imshow("Input Image", cv_image)
            cv2.waitKey(1)

        # Perform detection with YOLOv8 (segmentation)
        results = self.model.predict(cv_image)  

        # Check if masks are detected
        if not results or results[0].masks is None or len(results[0].masks.xy) == 0:
            self.get_logger().warn("⚠️ No lane segmentation detected.")
            return

        detections = results[0].masks.xy  # Contours of detected masks
        annotated_image = results[0].plot()  # YOLO-annotated image

        # Display YOLO results
        if os.getenv("DISPLAY") is not None:
            cv2.imshow("Processed Image", annotated_image)
            cv2.waitKey(1)

        # Publish the annotated image
        output_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        self.publisher.publish(output_msg)

        # Analyze detected lanes
        left_xs = []
        right_xs = []

        for mask in detections:  
            x_coords = np.array(mask)[:, 0]  # Extract X coordinates from the mask
            avg_x = np.mean(x_coords)  # Average X to determine left/right

            if avg_x < cv_image.shape[1] / 2:
                left_xs.append(avg_x)
            else:
                right_xs.append(avg_x)

        # Check if both lanes are detected
        if left_xs and right_xs:
            left_lane_x = np.mean(left_xs)
            right_lane_x = np.mean(right_xs)
            center_lane_x = (left_lane_x + right_lane_x) / 2

            image_center_x = cv_image.shape[1] / 2
            error = center_lane_x - image_center_x
            self.get_logger().info(f"🔹 Lateral error: {error}")

            self.control_car(error)
        else:
            self.get_logger().warn("⚠️ Only one lane detected. Check model training.")

    def control_car(self, error):
        # PID parameters
        Kp = 0.005  
        Ki = 0.0001  
        Kd = 0.001  

        # PID computation
        derivative = error - self.prev_error
        self.integral += error
        control = Kp * error + Ki * self.integral + Kd * derivative
        self.prev_error = error

        # Limit steering command
        max_steering = 0.3  
        steering_angle = max(-max_steering, min(max_steering, control))

        # Create the command message
        cmd = Twist()
        cmd.linear.x = 0.3  # Adjustable speed
        cmd.angular.z = float(-steering_angle)

        self.cmd_publisher.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    lane_detection_node = LaneDetectionNode()
    rclpy.spin(lane_detection_node)
    lane_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
