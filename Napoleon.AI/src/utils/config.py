import logging

# Global variables
MODEL_NAME = "deepseek-r1:32b"  # Can be changed as needed
BOT_NAME = "Napoleon"
MEMORY_FILE = "work_memory.pt"
EMBEDDING_DIM = 256
MEMORY_SIZE = 10

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ai_assistant")

# Check for required packages
required_packages = {
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