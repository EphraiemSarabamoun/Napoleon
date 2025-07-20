from typing import Optional, Any

from src.utils.config import logger, required_packages

class SpeechInterface:
    """Handles speech recognition and text-to-speech."""
    
    def __init__(self) -> None:
        self.tts_engine: Optional[Any] = None
        self.recognizer: Optional[Any] = None
        self.microphone: Optional[Any] = None
        
        # Initialize components if available
        self._init_tts()
        self._init_speech_recognition()
    
    def _init_tts(self) -> bool:
        """Initialize the text-to-speech engine."""
        if not required_packages["pyttsx3"]:
            return False
        
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            logger.info("Text-to-speech engine initialized")
            return True
        except Exception as e:
            logger.exception("Failed to initialize text-to-speech")
            return False
    
    def _init_speech_recognition(self) -> bool:
        """Initialize the speech recognition components."""
        if not required_packages["speech_recognition"]:
            return False
        
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            logger.info("Speech recognition initialized")
            return True
        except Exception as e:
            logger.exception("Failed to initialize speech recognition")
            return False
    
    def speak(self, text: str) -> bool:
        """Convert text to speech."""
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
        """Listen to voice input and return the recognized text."""
        if self.recognizer is None or self.microphone is None:
            logger.warning("Speech recognition not available")
            return None
        
        try:
            import speech_recognition as sr
            with self.microphone as source:
                logger.info("Listening...")
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=5)
            
            user_input = self.recognizer.recognize_google(audio)
            logger.info(f"Recognized: {user_input}")
            print(f"You: {user_input}")
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