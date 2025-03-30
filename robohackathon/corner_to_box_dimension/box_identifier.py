from aruco_detector import ArucoDetector
import numpy as np
class BoxIdentifier:
    def __init__(self,box_id,img_width, img_height):
        self.box_id = box_id
        self.aruco_detector = ArucoDetector(img_width=img_width,img_height=img_height, img_width=img_width)
    
    def tvec_to_angle(self,tvec):
        """
        Convert the translation vector to angles.
        """
        # Assuming tvec is a 3D vector [x, y, z]
        side_angle = np.arctan2(tvec[0], tvec[2])  # Angle in radians
        up_angle = np.arctan2(tvec[1], tvec[2])    # Angle in radians
        # Convert to degrees if needed
        side_angle_deg = np.degrees(side_angle)
        up_angle_deg = np.degrees(up_angle)
        return side_angle_deg, up_angle_deg


    def box_identify(self,img):
        """
        Identify the box in the image using ArUco markers.
        """
        detections = self.aruco_detector.detect(img)
        rvec = None
        tvec = None
        for detection in detections:
            if detection['id'] == self.box_id:
                # Extract the rotation and translation vectors
                rvec = np.array(detection['rvec'])
                tvec = np.array(detection['tvec'])
                # Convert to a more usable format if needed
        # print(f"Rotation Vector: {rvec}")
        # print(f"Translation Vector: {tvec}")
        # print("publishing ints (x,y):",tvec[0],tvec[1])
        if(tvec is not None):
            # Convert translation vector to angles
            side_angle, up_angle = self.tvec_to_angle(tvec)
            print("side angle:", side_angle)
            print("up angle:", up_angle)
        else:
            print(f"Box with ID {self.box_id} not detected.")

        
        raise NotImplementedError("Box identification logic not implemented.")