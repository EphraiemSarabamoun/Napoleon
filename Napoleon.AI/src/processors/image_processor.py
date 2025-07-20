import cv2
import numpy as np
import time
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from typing import List, Optional, Any

from src.utils.config import logger, required_packages

class ImageProcessor:
    """Handles image capture, classification, and OCR."""
    
    def __init__(self) -> None:
        self.camera: Optional[cv2.VideoCapture] = None
        self.camera_index: int = 0
        self.available_cameras: List[int] = []
        self.image_model: Optional[torch.nn.Module] = None
        self.preprocess: Optional[transforms.Compose] = None
        self.class_names: List[str] = []
        # Initialize components
        self._init_camera()
        self._init_image_model()
    
    def _init_camera(self) -> bool:
        """Initialize and find available cameras."""
        logger.info("Checking available cameras...")
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            logger.info(f"Camera index {0} is available.")
            self.camera_index = 0
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            return True
        else:
            logger.warning(f"Camera index {0} is not available.")
            return False
    
    def _init_image_model(self) -> bool:
        """Initialize the image classification model."""
        try:
            from torchvision.models import ResNet18_Weights
            weights = models.ResNet18_Weights.DEFAULT
            self.image_model = models.resnet18(weights=weights)
            self.image_model.eval()
            
            self.preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
            ])
             # Get human-readable class labels from weights metadata
            self.class_names = weights.meta["categories"]
            logger.info("Image model initialized successfully")
            return True
        except Exception as e:
            logger.exception("Failed to initialize image model")
            return False
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture an image from the camera."""
        if self.camera is None:
            logger.error("Camera not initialized")
            return None
        # Optionally set resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(2)  # Allow camera to warm up
        # Warm up the camera
        ret = False
        frame = None
        for _ in range(5):
            logger.info(self.camera.read())
            ret, frame = self.camera.read()
            if ret:
                cv2.imshow("Frame", frame)
                cv2.waitKey(0)
                break
            time.sleep(2)
        
        if not ret or frame is None:
            logger.error("Failed to capture image from camera")
            return None
        
        return frame
    
    def analyze_image(self, image: np.ndarray) -> int:
        """
        Analyze the captured image using the pre-trained model.
        Converts the image from BGR (OpenCV) to RGB before processing.
        """
        if self.image_model is None or self.preprocess is None:
            logger.error("Image model not initialized")
            return "Error: Image model not initialized"
        
        try:
            # Convert BGR to RGB for proper processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.preprocess(image_rgb)
            input_batch = input_tensor.unsqueeze(0)
            
            with torch.no_grad():
                output = self.image_model(input_batch)
            
            # Get the top prediction
            _, predicted_idx = torch.max(output, 1)
            predicted_idx = predicted_idx.item()
            predicted_label = (
                self.class_names[predicted_idx]
                if self.class_names and predicted_idx < len(self.class_names)
                else str(predicted_idx)
            )
            return predicted_label
        except Exception as e:
            logger.exception("Error analyzing image")
            return "Error analyzing image"
    
    def ocr_image(self, image: np.ndarray) -> str:
        """
        Performs OCR on the captured image.
        Converts the image to grayscale before processing to improve OCR results.
        """
        if not required_packages["pytesseract"]:
            return "OCR is not available (pytesseract not installed)"
        
        try:
            import pytesseract
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            return text.strip() if text else "No text detected"
        except Exception as e:
            logger.exception("Error performing OCR")
            return f"OCR error: {str(e)}"
    
    def release_resources(self) -> None:
        """Release camera resources."""
        if self.camera is not None:
            self.camera.release()
            logger.info("Camera resources released") 