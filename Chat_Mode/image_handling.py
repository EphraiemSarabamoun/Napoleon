import cv2
import easyocr
import os
import sys
import subprocess
import json
import datetime
import base64
import requests

# Fix encoding
sys.stdout.reconfigure(encoding='utf-8')

def analyze_image_with_llava(image_path, model_name="llava:13b"):
    """
    Send image to LLaVA model via Ollama for analysis.
    Returns the model's description of the image.
    """
    try:
        # Read and encode image to base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare payload for Ollama
        payload = {
            "model": model_name,
            "prompt": "Describe the content of this image in detail.",
            "images": [encoded_image]
        }
        
        # Call Ollama API
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        response.raise_for_status()
        
        # Parse response (Ollama's generate API streams responses line by line)
        description = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode('utf-8'))
                if "response" in data:
                    description += data["response"]
                if data.get("done", False):
                    break
        
        return description.strip() if description else "No description provided by LLaVA."
    
    except Exception as e:
        print(f"Error analyzing image with LLaVA: {e}")
        return f"Error analyzing image: {str(e)}"

def webcam_capture_and_analyze(save_dir="output_frames"):
    """
    Capture image from webcam, extract text with EasyOCR, and analyze with LLaVA.
    Returns a dictionary with extracted text and LLaVA description.
    """
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return {"error": "Cannot open camera. Please check webcam connection."}
    
    try:
        print("Camera opened successfully!")
        print("Taking picture in 3 seconds...")
        
        import time
        time.sleep(3)
        
        # Capture frame
        ret, frame = cap.read()
        
        if not ret:
            return {"error": "Failed to capture image"}
        
        # Save the captured image
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"webcam_capture_{timestamp}.jpg"
        image_path = os.path.join(save_dir, filename)
        cv2.imwrite(image_path, frame)
        
        print(f"Photo saved as: {image_path}")
        
        # Analyze image with LLaVA
        print("Analyzing image with LLaVA...")
        llava_description = analyze_image_with_llava(image_path)
        
        return {
            "image_path": image_path,
            "llava_description": llava_description
        }
    
    except Exception as e:
        return {"error": f"Error: {str(e)}"}
    
    finally:
        cap.release()

if __name__ == "__main__":
    result = webcam_capture_and_analyze()
    if "error" in result:
        print(result["error"])
        print("Make sure webcam is enabled and accessible.")
    else:
        print("Analysis complete!")
        print("Extracted Text:")
        print(result["extracted_text"])
        print("LLaVA Description:")
        print(result["llava_description"])