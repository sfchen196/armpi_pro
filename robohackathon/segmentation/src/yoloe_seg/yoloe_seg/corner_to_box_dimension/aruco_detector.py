import cv2
import numpy as np

class ArucoDetector:
    def __init__(self, img_width, img_height, dictionary_id=cv2.aruco.DICT_ARUCO_ORIGINAL):
        """
        Initialize the ArucoDetector.
        
        Parameters:
            marker_length (float): The physical side length of the marker (in meters).
            camera_matrix (numpy.ndarray): The camera intrinsic matrix.
            dist_coeffs (numpy.ndarray): The distortion coefficients.
            dictionary_id (int): The type of ArUco dictionary to use (default: cv2.aruco.DICT_4X4_50).
        """
        
        img_dim_orig = (640, 480)

        K_original = np.array([[519.8387366 ,   0.0, 336.09275031],
                      [  0.0, 519.18747867, 230.57932374],
                      [  0.0,   0.0,      1.0]])
        scale_x = img_width / img_dim_orig[0]
        scale_y = img_height / img_dim_orig[1]
        K = K_original.copy()
        K[0, 0] *= scale_x  # fx
        K[1, 1] *= scale_y  # fy
        K[0, 2] *= scale_x  # cx
        K[1, 2] *= scale_y  # cy
        # Example distortion coefficients.
        # Replace with your actual fisheye distortion coefficients.
        D = np.array([-0.12899328, -0.50262509, 1.30822224, -1.05831611])

        self.marker_length = 0.02
        self.camera_matrix = K
        self.dist_coeffs = D
        self.aruco_dict = self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = cv2.aruco.DetectorParameters()


    def detect(self, img):
        """
        Detect ArUco markers in an image and estimate their 3D orientation.
        
        Parameters:
            img (numpy.ndarray): The input image (BGR or grayscale).
            
        Returns:
            list of dict: A list of detections where each detection is a dictionary containing:
                - 'id': The marker ID.
                - 'rvec': The rotation vector (3x1) as a list.
                - 'tvec': The translation vector (3x1) as a list.
                - 'euler_angles': The orientation in Euler angles (roll, pitch, yaw) in radians.
        """
        # Convert image to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Detect markers in the image
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        detections = []

        if ids is not None:
            # Estimate pose for each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, 
                                                                   self.camera_matrix, self.dist_coeffs)
            for i, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[i]
                tvec = tvecs[i]
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                # Compute Euler angles from rotation matrix
                sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
                singular = sy < 1e-6

                if not singular:
                    roll  = np.arctan2(R[2, 1], R[2, 2])
                    pitch = np.arctan2(-R[2, 0], sy)
                    yaw   = np.arctan2(R[1, 0], R[0, 0])
                else:
                    roll  = np.arctan2(-R[1, 2], R[1, 1])
                    pitch = np.arctan2(-R[2, 0], sy)
                    yaw   = 0

                detection = {
                    'id': int(marker_id),
                    'rvec': rvec.flatten().tolist(),
                    'tvec': tvec.flatten().tolist(),
                    'euler_angles': [roll, pitch, yaw]
                }
                detections.append(detection)

        return detections

# Example usage:
if __name__ == "__main__":
    # Example camera calibration parameters (replace these with your actual calibration data)
    # img_pth = "images/active_image.jpg"
    img_pth= "webcam_imgs/6.jpg"
    # Read the image to get its dimensions
    img = cv2.imread(img_pth)
    img_width, img_height = img.shape[1], img.shape[0]

    # Initialize the detector
    detector = ArucoDetector(img_width, img_height)

    
    # Detect markers and print the results
    detections = detector.detect(img)
    # print("number of markers detected:", len(detections))
    for det in detections:
        print("image path:", img_pth)
        # print(f"Marker ID: {det['id']}")
        # print(f"Rotation Vector: {det['rvec']}")
        print(f"Translation Vector: {det['tvec']}")
        # print(f"Euler Angles (roll, pitch, yaw): {det['euler_angles']}\n")
