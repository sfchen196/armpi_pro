import rclpy
from rclpy.node import Node
import socket
import threading
import struct
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from corner_to_box_dimension.box_identifier import BoxIdentifier


class Boxidentifier(Node):
    def __init__(self):
        super().__init__('dummy_service_server')
        self.port = 5000  # Must match the port used by the ROS1 client
        # Start the socket server in a separate thread so the ROS node can spin.
        self.server_thread = threading.Thread(target=self.start_server, daemon=True)
        self.server_thread.start()

        self.bridge = CvBridge()
        self.box_identifier = BoxIdentifier(box_id=0, img_width=640, img_height=480)
    
        self.subscription = self.create_subscription(
                Image,
                'bridge_image',
                self.image_callback,
                10
            )

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_image = cv_image
            # print("Received image from ROS1 client")  
            
            # Optional: log only once
            self.get_logger().debug("Received and stored a new image.")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def start_server(self):
        # Create a TCP/IP socket.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.port))
            s.listen()
            self.get_logger().info(f"Socket server listening on port {self.port}")
            while rclpy.ok():
                conn, addr = s.accept()
                self.get_logger().info(f"Accepted connection from {addr}")
                # Handle each client connection in a separate thread.
                client_thread = threading.Thread(target=self.handle_client, args=(conn,), daemon=True)
                client_thread.start()

    def handle_client(self, conn):
        with conn:
            try:
                # Receive 4 bytes that represent an integer
                data = conn.recv(4)
                if not data or len(data) < 4:
                    self.get_logger().warn("No data (or incomplete data) received from client.")
                    return

                # Interpret the received bytes as an integer (signed, 32-bit)
                int_value = struct.unpack('i', data)[0]
                self.get_logger().info(f"Received int: {int_value}")
                self.box_identifier.box_id = int_value
                side = 0.0
                up=0.0
                for i in range(10):
                    side,up = self.box_identifier.box_identify(self.latest_image)
                    if side is not None and up is not None:
                        break
                    
                if side is None or up is None:
                    side = 0.0
                    up = 0.0

                # Here we define two dummy float values
                float1 = side
                float2 = up

                # Pack the floats into binary data (using 'ff' for two floats)
                response_data = struct.pack('ff', float1, float2)
                conn.sendall(response_data)
                self.get_logger().info(f"Sent response floats: {float1}, {float2}")

            except Exception as e:
                self.get_logger().error(f"Error handling client: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Boxidentifier()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
