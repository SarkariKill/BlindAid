import streamlit as st
import cv2
from ultralytics import YOLO
import threading

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8s.pt')

# Function to calculate distance in cm using the pinhole camera model
def calculate_distance_cm(focal_length, real_object_width, object_width_in_image):
    return (real_object_width * focal_length) / object_width_in_image

# Real object widths in centimeters (for common objects)
object_widths_cm = {
    'person': 45,          # Approximate width of a person in cm
    'bicycle': 60,         # Standard bicycle width
    'car': 180,            # Width of a car
    'motorbike': 80,       # Width of a motorbike
    'dog': 30,             # Width of a dog
    'chair': 50,           # Average chair width
    'table': 80,           # Dining table width
    'bottle': 7,           # Width of a bottle
    'backpack': 30,        # Width of a backpack
    'sofa': 200,           # Width of a large sofa
}

# Focal length in pixels (camera calibration)
focal_length = 700  # Focal length from your provided code

# Streamlit layout
st.title("Live YOLOv8 Object Detection with Distance Measurement")

# Checkbox to start/stop the camera
run = st.checkbox('Run Camera', value=False)

# Placeholder for displaying video
FRAME_WINDOW = st.empty()

# Set up a threading lock to safely manage camera access
lock = threading.Lock()

if run:
    st.write("Camera is running...")
    cap = cv2.VideoCapture(0)  # Open the camera feed

    # Set camera resolution (optional: lower resolution for better performance)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Camera calibration constants for distance calculation
    frame_skip = 3  # Process every 3rd frame to improve performance
    frame_count = 0

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Skip frames to reduce computation load
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # Perform object detection
        results = model(frame)

        # Extract bounding boxes and labels
        for result in results:
            for box in result.boxes:  # Iterate through detected objects
                label = model.names[int(box.cls)]  # Convert class index to label name
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

                # Calculate width of the detected object in the image
                object_width_in_image = x_max - x_min

                # Check if the object label has a known real-world width
                if label in object_widths_cm:
                    real_object_width = object_widths_cm[label]
                    # Calculate distance to the object using the pinhole camera model (in cm)
                    distance_cm = calculate_distance_cm(focal_length, real_object_width, object_width_in_image)

                    # Print only the distance to the object in the terminal
                    print(f'Distance to object: {distance_cm:.2f} cm')

                    # Display the bounding box and distance
                    # Set bounding box color to black (0, 0, 0)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)
                    text = f'Dist: {distance_cm:.2f} cm'
                    cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        FRAME_WINDOW.image(frame_rgb, channels="RGB")

    cap.release()  # Release the camera when done
else:
    st.write("Camera is stopped.")
