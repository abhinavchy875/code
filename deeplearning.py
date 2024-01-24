import cv2
import numpy as np
import tensorflow as tf

# Load pre-trained DeepLabv3 model
model = tf.keras.applications.DenseNet201(input_shape=(None, None, 3), include_top=False)
model.trainable = False

# Function to perform segmentation on a frame
def segment_frame(frame):
    # Resize frame to match the model's expected sizing
    input_tensor = tf.image.resize(frame, (256, 256))
    input_tensor = tf.expand_dims(input_tensor, 0)
    
    # Preprocess the input image
    input_tensor = tf.keras.applications.densenet.preprocess_input(input_tensor)
    
    # Get prediction from the model
    prediction = model.predict(input_tensor)
    
    # Post-process the output to get the segmented mask
    mask = np.argmax(prediction, axis=-1)
    
    return mask[0]

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Perform segmentation on the frame
    mask = segment_frame(frame)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=(mask == 15).astype(np.uint8) * 255)

    # Display the original frame and the segmented result
    cv2.imshow('Original', frame)
    cv2.imshow('Segmentation', result)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
