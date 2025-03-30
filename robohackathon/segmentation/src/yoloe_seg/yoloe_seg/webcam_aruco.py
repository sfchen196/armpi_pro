import cv2
import numpy as np
from aruco_detector import ArucoDetector


def main():
    # Open the webcam (device index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    # Read one frame to get image dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam.")
        return
    img_height, img_width = frame.shape[:2]

    # Instantiate the detector with the image dimensions
    detector = ArucoDetector(img_width, img_height)

    print("Starting webcam stream. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect markers in the current frame
        detections = detector.detect(frame)

        # Print detection results and draw axes
        for detection in detections:
            print("Marker ID:", detection['id'])
            print("Rotation Vector:", detection['rvec'])
            print("Translation Vector:", detection['tvec'])
            print("Euler Angles (roll, pitch, yaw):", detection['euler_angles'])
            print("------")

            # Draw the coordinate axes on the frame using cv2.drawFrameAxes
            rvec = np.array(detection['rvec']).reshape(3, 1)
            tvec = np.array(detection['tvec']).reshape(3, 1)
            cv2.drawFrameAxes(frame, detector.camera_matrix, detector.dist_coeffs,
                              rvec, tvec, detector.marker_length * 0.5, 2)

        # Display the frame
        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
