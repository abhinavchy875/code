import cv2
import numpy as np
from mrcnn import model as modellib
from mrcnn import visualize
# from mrcnn.config import Config

# Configuration for the Mask R-CNN model
class InferenceConfig(Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81  # 80 classes + 1 background (COCO dataset)

# Load the Mask R-CNN model
config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir="./")

# Download the pre-trained weights for COCO dataset
model.load_weights('mask_rcnn_coco.h5', by_name=True)

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the Mask R-CNN model to get predictions
    results = model.detect([frame_rgb], verbose=0)

    # Get the results for the first frame
    r = results[0]

    # Apply the mask to the original frame
    result = visualize.apply_mask(frame, r['masks'], (1, 1, 1), alpha=0.5)

    # Display the original frame and the segmented result
    cv2.imshow('Original', frame)
    cv2.imshow('Segmentation', result)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
