"""
MediaPipe Face Detection using the new Tasks API
This is compatible with mediapipe 0.10.x and Python 3.13+
"""

import cv2
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions
from mediapipe import Image, ImageFormat

import os
import urllib.request

# Download the built-in face detector model if it doesn't exist
MODEL_PATH = 'blaze_face_short_range.tflite'
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite'

if not os.path.exists(MODEL_PATH):
    print(f"Downloading {MODEL_PATH}...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

base_options = base_options_module.BaseOptions(model_asset_path=MODEL_PATH)
options = FaceDetectorOptions(base_options=base_options)

print("Face detector initialized successfully!")

# Open webcam
cap = cv2.VideoCapture(0)

with FaceDetector.create_from_options(options) as face_detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        
        # Detect faces
        detection_result = face_detector.detect(mp_image)

        # Draw bounding boxes
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x = int(bbox.origin_x)
                y = int(bbox.origin_y)
                width = int(bbox.width)
                height = int(bbox.height)
                
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()