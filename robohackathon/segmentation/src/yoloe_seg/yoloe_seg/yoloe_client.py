#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from yoloe_msgs.srv import DetectMasks  # Custom service: request contains an Image, response returns BoundingBoxes
import cv2
import numpy as np

class YOLOEClient(Node):
    def __init__(self):
        super().__init__('yoloe_client')
        self.cli = self.create_client(DetectMasks, 'detect_mask')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.bridge = CvBridge()

    def send_request(self, image_path):
        # Read the image from disk using OpenCV
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            self.get_logger().error(f"Failed to read image from {image_path}")
            return None

        # Convert the OpenCV image (BGR) to a ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        # Create and send the service request
        req = DetectMasks.Request()
        req.image = img_msg
        self.future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    client = YOLOEClient()
    image_path = "IMG_7470.jpeg"  # Update with the path to your saved image
    response = client.send_request(image_path)
    if response is not None:
        print("Received bounding boxes:")
        # The response contains a BoundingBoxes message which itself holds a list of BoundingBox messages in its 'boxes' field.
        mask_number =0
        for mask in response.masks.masks:
            print(f"rows: {mask.rows}, cols: {mask.cols}")
            # print(f"mask: {mask.data}")
            mask_np = np.array(mask.data, dtype=np.uint8).reshape((mask.rows, mask.cols))
            mask_image = np.zeros((mask.rows, mask.cols), dtype=np.uint8)

            # Convert 0/1 to boolean mask or rely on 1→True:
            mask_bool = mask_np.astype(bool)
            mask_image[mask_bool] = 255  # set those pixels to white
            cv2.imwrite(f"mask_{mask_number}.png", mask_image)
            mask_number += 1
            
        # for box in response.boxes.boxes:
        #     print(f"Class: {box.class_name}, Confidence: {box.confidence:.2f}, "
        #           f"Coordinates: ({box.xmin}, {box.ymin}) to ({box.xmax}, {box.ymax})")
    else:
        print("Service call failed.")
    client.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
