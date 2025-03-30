import rclpy
from rclpy.node import Node
import socket
import threading
import struct
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageBridgeServer(Node):
    def __init__(self):
        super().__init__('image_bridge_server')
        self.publisher_ = self.create_publisher(Image, 'bridge_image', 10)
        self.bridge = CvBridge()
        self.port = 6000  # Must match the ROS1 client port.
        # Start the socket server in a separate thread.
        self.server_thread = threading.Thread(target=self.start_server, daemon=True)
        self.server_thread.start()
        self.get_logger().info("Image bridge server started, waiting for connections...")

    def start_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.port))
            s.listen()
            self.get_logger().info("Listening on port {}".format(self.port))
            while rclpy.ok():
                conn, addr = s.accept()
                self.get_logger().info("Accepted connection from {}".format(addr))
                threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()

    def handle_client(self, conn):
        with conn:
            try:
                # Read 4 bytes for the length prefix.
                length_data = self.recvall(conn, 4)
                if not length_data:
                    self.get_logger().warn("Did not receive length prefix")
                    return
                length = struct.unpack('!I', length_data)[0]
                self.get_logger().info("Expecting image of length: {} bytes".format(length))
                # Read the JPEG data.
                image_data = self.recvall(conn, length)
                if not image_data:
                    self.get_logger().warn("Did not receive image data")
                    return
                self.get_logger().info("Received image data, actual length: {} bytes".format(len(image_data)))
                # Convert the byte data to a NumPy array and decode the JPEG.
                nparr = np.frombuffer(image_data, np.uint8)
                cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                # save the image to a file for debugging
                # cv2.imwrite('received_image.jpg', cv_image)
                if cv_image is None:
                    self.get_logger().error("Failed to decode image")
                    return
                # Convert OpenCV image to a ROS Image message.
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                self.publisher_.publish(ros_image)
                self.get_logger().info("Published image from ROS1 client")
            except Exception as e:
                self.get_logger().error("Error handling client: {}".format(e))

    def recvall(self, conn, n):
        data = b''
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data

def main(args=None):
    rclpy.init(args=args)
    node = ImageBridgeServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
