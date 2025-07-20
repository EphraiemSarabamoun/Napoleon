import cv2
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=False if CUDA not available

# Open camera (default camera is usually 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Optional: Resize the frame for faster processing
    resized_frame = cv2.resize(frame, (640, 480))

    # Perform OCR on the resized frame
    results = reader.readtext(resized_frame)

    # Draw bounding boxes and text on the frame
    for bbox, text, confidence in results:
        top_left = tuple([int(val) for val in bbox[0]])
        bottom_right = tuple([int(val) for val in bbox[2]])
        cv2.rectangle(resized_frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(resized_frame, text, (top_left[0], top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Real-time OCR', resized_frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()