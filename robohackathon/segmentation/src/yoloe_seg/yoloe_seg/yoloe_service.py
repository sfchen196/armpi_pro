#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# Import your custom service and message types.
# The service is assumed to have an 'image' field in the request and a 'boxes' field in the response.
# from yoloe_msgs.srv import DetectObjects
from yoloe_msgs.srv import DetectMasks 
from yoloe_msgs.msg import BoundingBox, BoundingBoxes

import torch
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from huggingface_hub import hf_hub_download
import supervision as sv
from yoloe_msgs.msg import MaskArray
from yoloe_msgs.msg import Mask

class YOLOEServiceNode(Node):
    def __init__(self):
        super().__init__('yoloe_service_node')
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

        # Create the service server named 'detect_objects'
        self.srv = self.create_service(DetectMasks, 'detect_mask', self.detect_objects_callback)
        self.get_logger().info('YOLOE Service Node started with fixed text prompt: "brown boxes"')

    def init_model(self, model_id):
        # Load the segmentation model from Hugging Face Hub
        filename = f"{model_id}-seg.pt"
        path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
        model = YOLOE(path)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Set fixed text prompt (change to "bags" or any desired prompt)
        fixed_prompt = ["brown boxes"]
        model.set_classes(fixed_prompt, model.get_text_pe(fixed_prompt))
        return model

    @smart_inference_mode()
    def detect_objects_callback(self, request, response):
        try:
            # Convert the incoming sensor_msgs/Image to an OpenCV image (BGR encoding)
            cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding="bgr8")
            # Convert the OpenCV image to a PIL image (RGB)
            pil_image = PILImage.fromarray(cv_image[:, :, ::-1])

            # Run inference using the YOLOE model
            results = self.model.predict(
                source=pil_image,
                imgsz=self.image_size,
                conf=self.conf_thresh,
                iou=self.iou_thresh
            )
            # Convert the inference results into detections using supervision
            detections = sv.Detections.from_ultralytics(results[0])
            masks = detections.mask
            masks_array = MaskArray()
            masks_array.masks = []
            for i in range(len(masks)):
                # If supervision returns a NumPy array, you're good; if Torch, call .cpu().numpy()
                mask_array = np.array(masks[i])      # shape: (height, width), float in [0..1]
                # Threshold it
                mask_bin = (mask_array > 0.5).astype(np.uint8)
                # Flatten
                mask_list = mask_bin.flatten().tolist()
                # Convert to booleans
                mask_boolean_list = [bool(x) for x in mask_list]

                mask_msg = Mask()
                mask_msg.data = mask_boolean_list
                mask_msg.rows = mask_bin.shape[0]
                mask_msg.cols = mask_bin.shape[1]
                masks_array.masks.append(mask_msg)

            response.masks = masks_array

            
            # # Create a bounding boxes message to store the results
            # bbox_msg = BoundingBoxes()
            # # Optionally, you can set a header with timestamp here if needed

            # # Extract each detection's bounding box and details
            # for xyxy, conf, cls_name in zip(detections.xyxy, detections.confidence, detections['class_name']):
            #     box = BoundingBox()
            #     box.class_name = cls_name
            #     box.confidence = float(conf)
            #     box.xmin = int(xyxy[0])
            #     box.ymin = int(xyxy[1])
            #     box.xmax = int(xyxy[2])
            #     box.ymax = int(xyxy[3])
            #     bbox_msg.boxes.append(box)

            # response.boxes = bbox_msg
            self.get_logger().info(f"Processed image, found {len(masks)} masks.")
        except Exception as e:
            self.get_logger().error(f"Detection failed: {e}")
        return response

def main(args=None):
    rclpy.init(args=args)
    node = YOLOEServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
