import cv2
import easyocr

# Initialize reader
reader = easyocr.Reader(['en'])

# Open video
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process every 30th frame (you can adjust as needed)
    if frame_number % 30 == 0:
        results = reader.readtext(frame)
        print(f'Frame {frame_number}:')
        for bbox, text, confidence in results:
            print(f'  Text: {text}, Confidence: {confidence:.2f}')

    frame_number += 1

cap.release()