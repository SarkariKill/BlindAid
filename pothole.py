import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('best.h5')

# Initialize webcam video feed
cap = cv2.VideoCapture(0)  # 0 for the default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to match model input size
    resize = cv2.resize(frame, (256, 256))

    # Convert frame to RGB for correct model input format
    resize_rgb = cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)

    # Normalize the frame to be in the range [0, 1]
    resize_norm = resize_rgb / 255.0

    # Make prediction
    yhat = model.predict(np.expand_dims(resize_norm, axis=0))

    # Check if pothole is detected
    if yhat > 0.5:
        label = 'Pothole Detected'
        color = (0, 0, 255)  # Red color for pothole
    else:
        label = 'Smooth Road'
        color = (0, 255, 0)  # Green color for smooth road

    # Display the label on the video frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the resulting frame
    cv2.imshow('Road Pothole Detection', frame)

    # Exit the video loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
