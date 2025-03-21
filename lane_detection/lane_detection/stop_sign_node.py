import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import os
import numpy as np
from ultralytics import YOLO
from geometry_msgs.msg import Twist

class StopSignNode(Node):
    def __init__(self):
        super().__init__('stop_sign_node')

        self.bridge = CvBridge()

        # Load the YOLOv8 model
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models', 'best_stop.pt')

        if not os.path.exists(model_path):
            self.get_logger().error(f"❌ STOP sign model not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.get_logger().info(f"✅ Loading YOLO STOP sign model...")
        self.model = YOLO(model_path)  # Remove torch.jit.load()

        # Subscribe to the image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  # Camera topic
            self.image_callback,
            10)
        
        # Publishers for the detected image and movement command
        self.publisher = self.create_publisher(Image, 'stop_sign/output', 10)
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Inference with YOLO (no need to transform to tensor)
        results = self.model.predict(cv_image)

        stop_detected = False
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of the bounding box
                score = float(box.conf[0])  # Confidence score
                label = int(box.cls[0])  # Predicted class (0 = STOP)

                if label == 0 and score > 0.5:  # Detection with confidence > 50%
                    stop_detected = True
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(cv_image, f"STOP {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if stop_detected:
            self.get_logger().info("🛑 STOP sign detected!")
        else:
            self.get_logger().info("🚫 No STOP sign detected.")

        # Display the annotated image
        if os.getenv("DISPLAY") is not None:
            cv2.imshow("STOP Sign Detection", cv_image)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = StopSignNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
