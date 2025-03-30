#!#!/home/abivishaq/projects/robohackathon/ros2_ws/venv/bin/python
import sys
print("Using Python interpreter:", sys.executable)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from yoloe_msgs.msg import BoundingBox, BoundingBoxes

import torch
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from huggingface_hub import hf_hub_download
import supervision as sv

class YOLOENode(Node):
    def __init__(self):
        super().__init__('yoloe_node')
        self.declare_parameter('model_id', 'yoloe-v8l')
        self.declare_parameter('image_size', 640)
        self.declare_parameter('conf_thresh', 0.25)
        self.declare_parameter('iou_thresh', 0.7)

        self.model_id = self.get_parameter('model_id').value
        self.image_size = self.get_parameter('image_size').value
        self.conf_thresh = self.get_parameter('conf_thresh').value
        self.iou_thresh = self.get_parameter('iou_thresh').value

        self.bridge = CvBridge()
        self.model = self.init_model(self.model_id)

        # Subscribe to the input image topic
        self.subscription = self.create_subscription(
            Image,
            '/input/image',
            self.image_callback,
            10
        )
        # Publisher for bounding boxes
        self.publisher = self.create_publisher(BoundingBoxes, '/output/bounding_boxes', 10)
        self.get_logger().info('YOLOE ROS2 Node started with fixed text prompt: "brown boxes"')

    def init_model(self, model_id):
        # Load the standard segmentation model
        filename = f"{model_id}-seg.pt"
        path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
        model = YOLOE(path)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Set fixed text prompt: "brown boxes"
        fixed_prompt = ["bags"]
        model.set_classes(fixed_prompt, model.get_text_pe(fixed_prompt))
        return model

    @smart_inference_mode()
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image (BGR encoding)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # Convert OpenCV image (BGR) to a PIL image (RGB)
            pil_image = PILImage.fromarray(cv_image[:, :, ::-1])

            # Run model inference
            results = self.model.predict(
                source=pil_image,
                imgsz=self.image_size,
                conf=self.conf_thresh,
                iou=self.iou_thresh
            )
            # Convert the inference results into detections using supervision
            detections = sv.Detections.from_ultralytics(results[0])

            # Create a bounding boxes message
            bbox_msg = BoundingBoxes()
            # (Optional) set header with timestamp if needed:
            # bbox_msg.header = Header()
            # bbox_msg.header.stamp = self.get_clock().now().to_msg()

            # Extract each detection's bounding box and details
            for xyxy, conf, cls_name in zip(detections.xyxy, detections.confidence, detections['class_name']):
                box = BoundingBox()
                box.class_name = cls_name
                box.confidence = float(conf)
                box.xmin = int(xyxy[0])
                box.ymin = int(xyxy[1])
                box.xmax = int(xyxy[2])
                box.ymax = int(xyxy[3])
                bbox_msg.boxes.append(box)

            # Publish the bounding boxes message
            self.publisher.publish(bbox_msg)
            self.get_logger().info(f"Published {len(bbox_msg.boxes)} bounding boxes.")

            # Annotate the image using supervision and save the result image
            resolution_wh = pil_image.size
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)
            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(detections['class_name'], detections.confidence)
            ]
            annotated_image = pil_image.copy()
            annotated_image = sv.MaskAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                opacity=0.4
            ).annotate(scene=annotated_image, detections=detections)
            annotated_image = sv.BoxAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                thickness=thickness
            ).annotate(scene=annotated_image, detections=detections)
            annotated_image = sv.LabelAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                text_scale=text_scale,
                smart_position=True
            ).annotate(scene=annotated_image, detections=detections, labels=labels)

            # Save the annotated image to disk, overwriting the same file each time
            annotated_image.save("result.jpg")
            self.get_logger().info("Saved result image as result.jpg")

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLOENode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
