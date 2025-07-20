import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

import subprocess
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import hashlib
import numpy as np
import re
import cv2
import logging
import time  # For camera warm-up
from typing import List, Tuple, Dict, Optional, Any

# New model options for reasoning vs non-reasoning
MODEL_NAME_NON_REASONING = "deepseek-r1:32b"
MODEL_NAME_REASONING = "deepseek-r1:32b_reasoning"
BOT_NAME = "Napoleon"
MEMORY_FILE = "work_memory.pt"
EMBEDDING_DIM = 256
MEMORY_SIZE = 10

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ai_assistant")

# Check for required packages
required_packages: Dict[str, bool] = {
    "pyttsx3": False,
    "speech_recognition": False,
    "pytesseract": False
}

try:
    import pyttsx3
    required_packages["pyttsx3"] = True
    logger.info("pyttsx3 is available")
except ImportError:
    logger.warning("pyttsx3 is not installed. Text-to-speech will be disabled.")

try:
    import speech_recognition as sr
    required_packages["speech_recognition"] = True
    logger.info("speech_recognition is available")
except ImportError:
    logger.warning("speech_recognition is not installed. Voice input will be disabled.")

try:
    import pytesseract
    required_packages["pytesseract"] = True
    logger.info("pytesseract is available")
except ImportError:
    logger.warning("pytesseract is not installed. OCR will be disabled.")


class ImageProcessor:
    """Handles image capture, classification, and OCR."""
    
    def __init__(self) -> None:
        self.camera: Optional[cv2.VideoCapture] = None
        self.camera_index: int = 0
        self.available_cameras: List[int] = []
        self.image_model: Optional[torch.nn.Module] = None
        self.preprocess: Optional[transforms.Compose] = None
        self.class_names: List[str] = []
        self._init_camera()
        self._init_image_model()
    
    def _init_camera(self) -> bool:
        logger.info("Checking available cameras...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            logger.info("Camera index 0 is available.")
            self.camera_index = 0
            self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            return True
        else:
            logger.warning("Camera index 0 is not available.")
            return False
        
    def _init_image_model(self) -> bool:
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
            self.class_names = weights.meta["categories"]
            logger.info("Image model initialized successfully")
            return True
        except Exception as e:
            logger.exception("Failed to initialize image model")
            return False
    
    def capture_image(self) -> Optional[np.ndarray]:
        if self.camera is None:
            logger.error("Camera not initialized")
            return None
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(2)  # Allow camera to warm up
        ret, frame = None, None
        for _ in range(5):
            ret, frame = self.camera.read()
            if ret:
                # For API mode, we avoid opening a display window.
                break
            time.sleep(2)
        
        if not ret or frame is None:
            logger.error("Failed to capture image from camera")
            return None
        
        return frame
    
    def analyze_image(self, image: np.ndarray) -> str:
        if self.image_model is None or self.preprocess is None:
            logger.error("Image model not initialized")
            return "Error: Image model not initialized"
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.preprocess(image_rgb)
            input_batch = input_tensor.unsqueeze(0)
            with torch.no_grad():
                output = self.image_model(input_batch)
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
        if not required_packages["pytesseract"]:
            return "OCR is not available (pytesseract not installed)"
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            return text.strip() if text else "No text detected"
        except Exception as e:
            logger.exception("Error performing OCR")
            return f"OCR error: {str(e)}"
    
    def release_resources(self) -> None:
        if self.camera is not None:
            self.camera.release()
            logger.info("Camera resources released")


class SpeechInterface:
    """Handles speech recognition and text-to-speech."""
    
    def __init__(self) -> None:
        self.tts_engine: Optional[Any] = None
        self.recognizer: Optional[sr.Recognizer] = None
        self.microphone: Optional[sr.Microphone] = None
        self._init_tts()
        self._init_speech_recognition()
    
    def _init_tts(self) -> bool:
        if not required_packages["pyttsx3"]:
            return False
        try:
            self.tts_engine = pyttsx3.init()
            logger.info("Text-to-speech engine initialized")
            return True
        except Exception as e:
            logger.exception("Failed to initialize text-to-speech")
            return False
    
    def _init_speech_recognition(self) -> bool:
        if not required_packages["speech_recognition"]:
            return False
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            logger.info("Speech recognition initialized")
            return True
        except Exception as e:
            logger.exception("Failed to initialize speech recognition")
            return False
    
    def speak(self, text: str) -> bool:
        if self.tts_engine is None:
            logger.warning("Text-to-speech not available")
            return False
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            return True
        except Exception as e:
            logger.exception("Error in text-to-speech")
            return False
    
    def listen(self) -> Optional[str]:
        if self.recognizer is None or self.microphone is None:
            logger.warning("Speech recognition not available")
            return None
        try:
            with self.microphone as source:
                logger.info("Listening...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5)
            user_input = self.recognizer.recognize_google(audio)
            logger.info(f"Recognized: {user_input}")
            return user_input
        except sr.WaitTimeoutError:
            logger.warning("Listening timed out - no speech detected")
            self.speak("I didn't hear anything")
            return None
        except sr.UnknownValueError:
            logger.warning("Speech not understood")
            self.speak("Sorry, I did not understand that")
            return None
        except sr.RequestError as e:
            logger.exception("Google Speech Recognition service error")
            self.speak("Could not request results from Google Speech Recognition service")
            return None
        except Exception as e:
            logger.exception("Error in speech recognition")
            return None


class TextProcessor:
    """Handles text processing and name extraction."""
    
    @staticmethod
    def clean_response(response_text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        return cleaned.strip()
    
    @staticmethod
    def extract_user_name(text: str) -> Optional[str]:
        pattern = re.compile(r"my name is\s+([\w\s]+)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        return None
    
    @staticmethod
    def get_user_name_from_entries(entries: List[str], bot_name: str) -> Optional[str]:
        for entry in reversed(entries):
            name = TextProcessor.extract_user_name(entry)
            if name and name.lower() != bot_name.lower():
                return name
        return None


class PersistentMemory(torch.nn.Module):
    """Manages persistent memory for the AI assistant."""
    
    def __init__(self, memory_file: str, memory_size: int = 10, embedding_dim: int = 256) -> None:
        super(PersistentMemory, self).__init__()
        self.memory_file: str = memory_file
        self.memory_size: int = memory_size
        self.embedding_dim: int = embedding_dim
        
        self.register_buffer('memory', torch.zeros(memory_size, embedding_dim))
        self.register_buffer('usage', torch.zeros(memory_size))
        self.entries: List[str] = []
        
        if os.path.exists(self.memory_file):
            self.load_memory()
            logger.info(f"Loaded memory from {self.memory_file}")
        else:
            logger.info("Starting with a new memory")
    
    def _fallback_embedding(self, text: str) -> torch.Tensor:
        logger.info("Using fallback embedding method")
        h = hashlib.sha256(text.encode('utf-8')).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
        if len(v) < self.embedding_dim:
            v = np.pad(v, (0, self.embedding_dim - len(v)), mode='constant')
        else:
            v = v[:self.embedding_dim]
        return torch.tensor(v, dtype=torch.float)
    
    def write(self, text: str) -> int:
        embedding = self.embed_text(text)
        usage_scores = -self.usage  # lower usage gives higher score
        probabilities = F.softmax(usage_scores, dim=0)
        slot_index = torch.argmax(probabilities).item()
        self.memory[slot_index] = embedding
        self.usage.mul_(0.9)
        self.usage[slot_index] = 1.0
        
        if slot_index < len(self.entries):
            self.entries[slot_index] = text
        else:
            self.entries.append(text)
            
        return slot_index
    
    def read(self, query_text: str, top_k: int = 3) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        query_embedding = self.embed_text(query_text)
        memory_norm = self.memory.norm(dim=1, keepdim=True) + 1e-10
        query_norm = query_embedding.norm() + 1e-10
        normalized_memory = self.memory / memory_norm
        normalized_query = query_embedding / query_norm
        cosine_sim = torch.matmul(normalized_memory, normalized_query)
        top_k = min(top_k, len(self.entries), self.memory_size)
        topk_values, topk_indices = torch.topk(cosine_sim, top_k)
        retrieved_entries = []
        for idx in topk_indices.tolist():
            if idx < len(self.entries):
                retrieved_entries.append(self.entries[idx])
            else:
                retrieved_entries.append("[Empty Slot]")
        return topk_indices, topk_values, retrieved_entries
    
    def save_memory(self) -> bool:
        try:
            state = {
                "memory": self.memory,
                "usage": self.usage,
                "entries": self.entries
            }
            torch.save(state, self.memory_file)
            logger.info(f"Memory saved to {self.memory_file}")
            return True
        except Exception as e:
            logger.exception("Error saving memory")
            return False
    
    def load_memory(self) -> bool:
        try:
            state = torch.load(self.memory_file)
            self.memory.copy_(state["memory"])
            self.usage.copy_(state["usage"])
            self.entries = state["entries"]
            return True
        except Exception as e:
            logger.exception("Error loading memory")
            return False
    
def embed_text(self, text: str) -> torch.Tensor:
    try:
        command = ["ollama", "run", MODEL_NAME_NON_REASONING, text]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=300
        )
        output_text = result.stdout.strip()
        if not output_text:
            logger.warning(f"No output received from {MODEL_NAME_NON_REASONING}")
            return self._fallback_embedding(text)
        
        # Only try to parse as JSON if it appears to be JSON.
        if output_text.startswith("{") and output_text.endswith("}"):
            try:
                output_json = json.loads(output_text)
                if "embedding" in output_json:
                    embedding_list = output_json["embedding"]
                    if len(embedding_list) != self.embedding_dim:
                        logger.warning(f"Embedding dimension mismatch: got {len(embedding_list)}, expected {self.embedding_dim}")
                        if len(embedding_list) > self.embedding_dim:
                            embedding_list = embedding_list[:self.embedding_dim]
                        else:
                            embedding_list = embedding_list + [0.0] * (self.embedding_dim - len(embedding_list))
                    return torch.tensor(embedding_list, dtype=torch.float)
                else:
                    logger.warning("Response JSON does not contain embedding")
                    response_text = output_json.get("response", output_text)
                    return self._fallback_embedding(response_text)
            except json.JSONDecodeError:
                logger.warning("JSON decode error even though output looked like JSON")
                return self._fallback_embedding(output_text)
        else:
            logger.warning("Output is not in JSON format; using fallback embedding")
            return self._fallback_embedding(output_text)
                    
    except subprocess.CalledProcessError as e:
        logger.exception("Error obtaining embedding")
        return self._fallback_embedding(text)
    except subprocess.TimeoutExpired as e:
        logger.exception("Embedding generation timed out")
        return self._fallback_embedding(text)
    except Exception as e:
        logger.exception("Unexpected error obtaining embedding")
        return self._fallback_embedding(text)


class LLMInterface:
    """Handles interactions with the LLM."""
    
@staticmethod
def generate_response(prompt: str, model_name: str = MODEL_NAME_NON_REASONING, timeout: int = 120) -> str:
    modified_prompt = (
        prompt +
        "\nAnswer directly and concisely. " +
        "Do not provide a summary of the conversation; just answer the user's question."
    )
    
    command = ["ollama", "run", model_name, modified_prompt]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout  # Increased timeout to 120 seconds
        )
        output_text = result.stdout.strip()
        if not output_text:
            logger.warning(f"No output received from {model_name}")
            return "[No response generated]"
        
        # If the output appears to be JSON, attempt to parse it.
        if output_text.startswith("{") and output_text.endswith("}"):
            try:
                output_json = json.loads(output_text)
                response = output_json.get("response", "")
                if not response:
                    response = output_text
            except json.JSONDecodeError:
                response = output_text
        else:
            response = output_text
        
        return response
        
    except subprocess.CalledProcessError as e:
        logger.exception("Error generating response")
        return "[Error generating response]"
    except subprocess.TimeoutExpired as e:
        logger.exception("Response generation timed out")
        return "[Response generation timed out]"
    except Exception as e:
        logger.exception("Unexpected error generating response")
        return "[Error generating response]"


class AIAssistant:
    """Main class for the AI assistant with memory, speech, and image capabilities."""
    
    def __init__(
        self, 
        bot_name: str = BOT_NAME,
        memory_file: str = MEMORY_FILE,
        memory_size: int = MEMORY_SIZE,
        embedding_dim: int = EMBEDDING_DIM
    ) -> None:
        self.bot_name: str = bot_name
        self.user_name: Optional[str] = None
        
        self.memory = PersistentMemory(memory_file, memory_size, embedding_dim)
        self.speech = SpeechInterface()
        self.image = ImageProcessor()
        self.llm = LLMInterface()
        self.text_processor = TextProcessor()
    
    def process_special_commands(self, user_input: str) -> Optional[str]:
        if user_input.lower() in ["exit", "quit"]:
            return "EXIT"
        if re.search(r"what(?:'s|s| is) my name\??", user_input.lower()):
            return f"Your name is {self.user_name or 'not set'}."
        if re.search(r"what(?:'s|s| is) your name\??", user_input.lower()):
            return f"Hi! My name is {self.bot_name}!"
        if re.search(r"take a picture", user_input.lower()):
            frame = self.image.capture_image()
            if frame is not None:
                # Instead of displaying the image, perform OCR and return text.
                text = self.image.ocr_image(frame)
                return f"I have taken a picture. The extracted text is: {text}"
            else:
                return "Failed to take a picture."
        return None

    def process_query(
        self,
        user_input: str,
        input_mode: str = "text",
        output_mode: str = "text",
        show_reasoning: bool = False,
        model_option: str = "non_reasoning"
    ) -> dict:
        # Process any special commands
        special_response = self.process_special_commands(user_input)
        if special_response == "EXIT":
            return {"bot_response": "Exiting. Goodbye!"}
        
        # Write user input to memory and retrieve recent entries
        self.memory.write(user_input)
        _, _, entries = self.memory.read(user_input, top_k=3)
        conversation_history = "\n".join(self.memory.entries[-5:])
        
        # Adjust prompt and model based on model_option toggle
        if model_option == "reasoning":
            reasoning_instruction = "Explain your reasoning step-by-step. "
            model_name = MODEL_NAME_REASONING
        else:
            reasoning_instruction = ""
            model_name = MODEL_NAME_NON_REASONING
        
        prompt = (
            f"Conversation so far:\n{conversation_history}\n"
            f"User's question: '{user_input}'\n"
            f"{reasoning_instruction}"
            "Answer the question directly and concisely."
        )
        
        bot_response = self.llm.generate_response(prompt, model_name=model_name)
        # Remove chain-of-thought if the user chooses not to see it
        if not show_reasoning:
            bot_response = self.text_processor.clean_response(bot_response)
        
        if output_mode == "audio":
            self.speech.speak(bot_response)
        
        self.memory.save_memory()
        return {"bot_response": bot_response, "memory_entries": entries}
    
    def cleanup(self) -> None:
        self.memory.save_memory()
        self.image.release_resources()
        logger.info("Resources cleaned up")


# Create a global assistant instance so that state persists between API calls
assistant = AIAssistant()

@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json() or {}
    input_mode = data.get("input_mode", "text")
    output_mode = data.get("output_mode", "text")
    show_reasoning = data.get("show_reasoning", False)
    model_option = data.get("model_option", "non_reasoning")
    user_input = data.get("user_input", "")
    
    # If audio input mode and an audio file is provided, use speech recognition to extract text.
    if input_mode == "audio" and "audio" in request.files:
        if required_packages["speech_recognition"]:
            try:
                sr_instance = sr.Recognizer()
                with sr.AudioFile(request.files["audio"]) as source:
                    audio = sr_instance.record(source)
                user_input = sr_instance.recognize_google(audio)
            except Exception as e:
                logger.exception("Error processing audio input")
                return jsonify({"bot_response": "Error processing audio input."})
        else:
            return jsonify({"bot_response": "Speech recognition is not available."})
    
    response = assistant.process_query(
        user_input=user_input,
        input_mode=input_mode,
        output_mode=output_mode,
        show_reasoning=show_reasoning,
        model_option=model_option
    )
    return jsonify(response)


if __name__ == "__main__":
    try:
        # Run the Flask app
        app.run(host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.exception("Unexpected error")
    finally:
        assistant.cleanup()