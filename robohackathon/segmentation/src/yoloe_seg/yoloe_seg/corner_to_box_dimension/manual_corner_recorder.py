import cv2
import numpy as np

# Global variables for storing points.
# all_points: list of sets (each set is a list of 4 (x, y) tuples)
all_points = []
current_set = []  # Points of the current set (waiting for 4)

def click_event(event, x, y, flags, param):
    global all_points, current_set, annotated_img

    # When left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Record the point
        current_set.append((x, y))
        print(f"Clicked: {(x, y)}")
        
        # Draw a small circle on the image at the clicked point
        cv2.circle(annotated_img, (x, y), 5, (0, 0, 255), -1)
        # Optionally, label the point with its order in the current set
        cv2.putText(annotated_img, str(len(current_set)), (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Image', annotated_img)
        
        # When 4 points are collected, save the set and reset for the next set.
        if len(current_set) == 4:
            all_points.append(current_set.copy())
            print("Recorded set:", current_set)
            current_set = []

def main():
    global annotated_img
    
    # Load the image; update the filename if needed.
    image_file = 'active_image.png'
    img = cv2.imread(image_file)
    if img is None:
        print("Error: Unable to load image!")
        return

    # Create a copy of the image for drawing annotated points.
    annotated_img = img.copy()

    # Create a window and attach the mouse callback.
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', click_event)

    print("Click 4 points for each object set. Press 'q' to finish and save.")

    # Display loop
    while True:
        cv2.imshow('Image', annotated_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to quit.
            break

    cv2.destroyAllWindows()

    # Save the points into a text file.
    # Each line corresponds to a set of 4 points: "x1,y1 x2,y2 x3,y3 x4,y4"
    with open('points.txt', 'w') as f:
        for set_points in all_points:
            line = " ".join([f"{pt[0]},{pt[1]}" for pt in set_points])
            f.write(line + "\n")
    print("Saved points to points.txt")

    # Save the annotated image with drawn points.
    cv2.imwrite('annotated_image.jpg', annotated_img)
    print("Saved annotated image as annotated_image.jpg")

if __name__ == '__main__':
    main()
