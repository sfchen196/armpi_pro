import cv2
import numpy as np

class FisheyeToPinholeConverter:
    def __init__(self, K, D, dim, balance=0.0, new_dim=None):
        """
        Initializes the converter with camera calibration parameters.

        Parameters:
            K (np.ndarray): 3x3 intrinsic camera matrix for the fisheye camera.
            D (np.ndarray): Distortion coefficients (array of 4 elements).
            dim (tuple): (width, height) of the input fisheye image.
            balance (float): Trade-off between field-of-view and pixel retention [0,1].
                             0 means the undistorted image keeps only the valid pixels,
                             while 1 retains all the pixels (but may include black areas).
            new_dim (tuple, optional): Desired (width, height) for the output pinhole image.
                                       If None, the input dimension is used.
        """
        self.K = K
        self.D = D
        self.dim = dim
        self.new_dim = new_dim if new_dim is not None else dim

        # Compute new optimal camera matrix for undistortion.
        # This adjusts the field of view based on the balance parameter.
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, dim, np.eye(3), balance=balance
        )

        # Precompute the undistortion and rectification transformation map.
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), self.new_K, self.new_dim, cv2.CV_16SC2
        )

    def convert(self, fisheye_image):
        """
        Converts the provided fisheye image to a pinhole model image using
        the precomputed undistortion maps.

        Parameters:
            fisheye_image (np.ndarray): Input fisheye image.

        Returns:
            np.ndarray: Undistorted (pinhole model) image.
        """
        undistorted_img = cv2.remap(
            fisheye_image, self.map1, self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        return undistorted_img

# Example usage:
if __name__ == '__main__':
    # Load a fisheye image (replace 'fisheye.jpg' with your image file)
    fisheye_img = cv2.imread('IMG_4402.jpg')
    if fisheye_img is None:
        print("Error: Could not load the fisheye image.")
    else:
        fisheye_img = cv2.resize(fisheye_img, (640, 480))
        # Get the image dimensions (width, height)
        height, width = fisheye_img.shape[:2]

        # Example intrinsic matrix.
        # Replace these values with your actual calibration data.
        K = np.array([[519.8387366 ,   0.0, 336.09275031],
                      [  0.0, 519.18747867, 230.57932374],
                      [  0.0,   0.0,      1.0]])

        # Example distortion coefficients.
        # Replace with your actual fisheye distortion coefficients.
        D = np.array([-0.12899328, -0.50262509, 1.30822224, -1.05831611])

        # Instantiate the converter.
        # The balance parameter controls the trade-off between FOV and valid pixel area.
        print("image_width/2:", width/2)
        print("image_height/2:", height/2)
        converter = FisheyeToPinholeConverter(K, D, (width, height), balance=0.5)

        # Convert the fisheye image to a pinhole model image.
        pinhole_img = converter.convert(fisheye_img)

        # Display both the original fisheye image and the undistorted image.
        cv2.imshow('Fisheye Image', fisheye_img)
        cv2.imshow('Pinhole Model Image', pinhole_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
