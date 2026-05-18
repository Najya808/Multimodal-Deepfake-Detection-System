import os
import cv2
from PIL import Image

from backend.face_detector import detect_faces
from backend.face_classifier import predict_face

def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    results = []

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # analyze every 30th frame
        if frame_count % 30 == 0:

            image = Image.fromarray(frame)

            faces = detect_faces(image)

            for face in faces:

                result = predict_face(face)

                results.append(result)

        frame_count += 1

    cap.release()

    return results