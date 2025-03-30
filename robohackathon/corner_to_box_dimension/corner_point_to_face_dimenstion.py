# V1: Assume box is  at edge of rack hence depth is known
#    + the box is not rotated

import numpy as np
from matplotlib import pyplot as plt

class BoxFaceDetector:
    def __init__(self):
        pass
    def corner_point_arranger(self, corner_points):
        """
        Arrange corner points of a box in a specific order.
        """
        # Assuming corner_points is a list of tuples (x, y, z)
        # Sort the points based on some criteria
        corner_points.sort(key=lambda point: (-1*point[1], point[0]))
        return corner_points
    def corner_point_visualizer(self, corner_points):
        """
        Visualize the corner points of a box. by plotting on the image with numbers
        """
        img_pth = "active_image.png"
        img = plt.imread(img_pth)
        print("corner points are", corner_points)

        for i, point in enumerate(corner_points):
            plt.scatter(point[0], point[1], color='red')
            plt.text(point[0], point[1], str(i), fontsize=12, color='blue')
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def get_depth_of_shelf(self, img):
        """
        Get the depth of the shelf from the image. using aruco
        """
        # Placeholder for actual depth extraction logic
        # For now, we return a dummy value
        return 5.0
    def corner_point_to_face_dimension(self, corner_points,depth):
        """
        Convert corner points of a box to face dimensions."
        """
        


if __name__ == "__main__":
    box_face_detector = BoxFaceDetector()
    # real_corner_points = [(840,696),(673,679), (676,604), (840,612)]
    corner_points = [(367,586), (458,586), (458,646), (365,644)]
    arranged_points = box_face_detector.corner_point_arranger(corner_points)
    print(arranged_points)
    box_face_detector.corner_point_visualizer(arranged_points)
    


        