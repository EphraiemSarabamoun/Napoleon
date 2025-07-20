import cv2
import easyocr
import os
import time

# Create output directory if it doesn't exist
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_count = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Only process every 10th frame to improve performance
    frame_count += 1
    if frame_count % 10 == 0:
        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (640, 480))

        # Perform OCR on the resized frame
        try:
            results = reader.readtext(resized_frame)
            
            # Draw bounding boxes and text on the frame
            for bbox, text, confidence in results:
                if confidence > 0.2:  # Only show results with confidence above threshold
                    top_left = tuple([int(val) for val in bbox[0]])
                    bottom_right = tuple([int(val) for val in bbox[2]])
                    cv2.rectangle(resized_frame, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.putText(resized_frame, f"{text} ({confidence:.2f})", 
                                (top_left[0], top_left[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Save the frame with OCR results
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(output_path, resized_frame)
            print(f"Saved frame with OCR results to {output_path}")
            
            # Print detected text
            if results:
                print("Detected text:")
                for _, text, confidence in results:
                    if confidence > 0.2:
                        print(f"- '{text}' (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    # Break if 'q' is pressed (won't work without imshow, need to use keyboard interrupt)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
cap.release()
print("Camera released. Program finished.")