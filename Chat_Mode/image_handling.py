import cv2
import easyocr
import os
import sys

# Fix encoding
sys.stdout.reconfigure(encoding='utf-8')

def webcam_test():
    """Test webcam capture and text analysis"""
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera. Please check webcam connection.")
        return
    
    print("Camera opened successfully!")
    print("Taking picture in 3 seconds...")
    
    import time
    time.sleep(3)
    
    # Capture frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        cap.release()
        return
    
    # Save the captured image to output_frames directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"output_frames/webcam_capture_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    cap.release()
    
    print(f"Photo saved as: {filename}")
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], verbose=False)
    
    # Analyze the image
    print("Analyzing image for text...")
    results = reader.readtext(frame)
    
    if results:
        print(f"Found {len(results)} text regions:")
        for i, (bbox, text, confidence) in enumerate(results, 1):
            print(f"  {i}. '{text}' (confidence: {confidence:.2f})")
    else:
        print("No text detected in the image")
    
    print("Test complete!")

if __name__ == "__main__":
    try:
        webcam_test()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure webcam is enabled and accessible.")
