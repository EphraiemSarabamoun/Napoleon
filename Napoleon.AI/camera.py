import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera failed to open.")
else:
    # Optionally set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    time.sleep(2)  # Allow camera to warm up
    
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
    else:
        print("Failed to read a frame.")
    cap.release()
    cv2.destroyAllWindows()