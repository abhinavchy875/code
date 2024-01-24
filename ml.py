import cv2
import numpy as np

# Function to perform segmentation on a frame using GrabCut
def segment_frame(frame):
    # Create a mask initialized with zeros
    mask = np.zeros(frame.shape[:2], np.uint8)

    # Define rectangle for initial segmentation
    rect = (50, 50, frame.shape[1] - 50, frame.shape[0] - 50)

    # Initialize background and foreground models for GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify mask to create a binary mask for foreground
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    return mask2

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Perform segmentation on the frame
    mask = segment_frame(frame)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask * 255)

    # Display the original frame and the segmented result
    cv2.imshow('Original', frame)
    cv2.imshow('Segmentation', result)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
