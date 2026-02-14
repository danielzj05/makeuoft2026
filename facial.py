import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Setup path to the model bundle
model_path = 'face_landmarker.task' 

# 2. Configure the Landmarker Task
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5)

# 3. Start Webcam
cap = cv2.VideoCapture(0)

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Prepare image for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)

        # Run Detection
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 4. Manual Drawing (Since mp.solutions is gone)
        if result.face_landmarks:
            for face_landmarks in result.face_landmarks:
                for landmark in face_landmarks:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow('MakeUofT - Face Mesh', cv2.flip(frame, 1))

        if cv2.waitKey(5) & 0xFF == 27: # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()