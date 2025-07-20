import re
import cv2
from typing import Optional, List

from src.utils.config import MODEL_NAME, BOT_NAME, MEMORY_FILE, MEMORY_SIZE, EMBEDDING_DIM, logger, required_packages
from src.interfaces.speech_interface import SpeechInterface
from src.interfaces.llm_interface import LLMInterface
from src.processors.image_processor import ImageProcessor
from src.processors.text_processor import TextProcessor
from src.memory.persistent_memory import PersistentMemory

class AIAssistant:
    """Main class for the AI assistant with memory, speech, and image capabilities."""
    
    def __init__(
        self, 
        model_name: str = MODEL_NAME,
        bot_name: str = BOT_NAME,
        memory_file: str = MEMORY_FILE,
        memory_size: int = MEMORY_SIZE,
        embedding_dim: int = EMBEDDING_DIM
    ) -> None:
        self.model_name: str = model_name
        self.bot_name: str = bot_name
        self.user_name: Optional[str] = None
        
        self.memory = PersistentMemory(memory_file, memory_size, embedding_dim)
        self.speech = SpeechInterface()
        self.image = ImageProcessor()
        self.llm = LLMInterface()
        self.text_processor = TextProcessor()
    
    def get_user_input(self) -> Optional[str]:
        """Get user input via voice (if available) or text."""
        if required_packages["speech_recognition"]:
            user_input = self.speech.listen()
            if user_input:
                return user_input
                
        try:
            user_input = input("\nYou (text): ").strip()
            return user_input
        except Exception as e:
            logger.exception("Error getting text input")
            return None
    
    def respond(self, text: str) -> None:
        """Respond to the user via voice and text."""
        print(f"\n{self.bot_name}: {text}")
        self.speech.speak(text)
    
    def initialize_user(self) -> bool:
        """Initialize user interaction by retrieving or asking for the user's name."""
        stored_name = TextProcessor.get_user_name_from_entries(
            self.memory.entries, bot_name=self.bot_name
        )
        
        if stored_name:
            self.user_name = stored_name
            logger.info(f"Retrieved user name from memory: {self.user_name}")
            return True
        
        initial_message = f"My name is {self.bot_name}. What's your name?"
        self.respond(initial_message)
        self.memory.write(initial_message)
        
        user_input = self.get_user_input()
        if not user_input:
            return False
        
        extracted_name = TextProcessor.extract_user_name(user_input)
        if not extracted_name:
            extracted_name = user_input  # Fallback if regex doesn't match
        
        self.user_name = extracted_name
        self.memory.write(f"My name is {self.user_name}")
        logger.info(f"New user name: {self.user_name}")
        
        welcome_message = f"Nice to meet you, {self.user_name}!"
        self.respond(welcome_message)
        self.memory.save_memory()
        
        return True
    
    def process_special_commands(self, user_input: str) -> Optional[str]:
        """
        Process special commands and queries.
        Returns a response if a special command is detected, or None otherwise.
        """
        if user_input.lower() in ["exit", "quit"]:
            self.respond("Exiting. Goodbye!")
            return "EXIT"
        
        if re.search(r"what(?:'s|s| is) my name\??", user_input.lower()):
            return f"Your name is {self.user_name}."
            
        if re.search(r"what(?:'s|s| is) your name\??", user_input.lower()):
            return f"Hi! My name is {self.bot_name}!"
        
        if re.search(r"take a picture", user_input.lower()):
            frame = self.image.capture_image()
            if frame is not None:
                cv2.imshow("Captured Image", frame)
                cv2.waitKey(1000)  # Display for 1 second
                cv2.destroyAllWindows()
                
                text = self.image.ocr_image(frame)
                return f"I have taken a picture. The extracted text is: {text}"
            else:
                return "Failed to take a picture."
        
        return None

    def run(self) -> None:
        """Run the main interaction loop."""
        logger.info("Starting AI assistant")
        print(f"\nWelcome to the AI assistant using {self.model_name}.")
        print("You can speak or type. Say 'exit' to quit.")
        
        if not self.initialize_user():
            logger.error("Failed to initialize user interaction")
            return
        
        while True:
            # Get user input
            user_input = self.get_user_input()
            if not user_input:
                continue
            
            # Process special commands (exit, name queries, etc.)
            special_response = self.process_special_commands(user_input)
            if special_response == "EXIT":
                break
            
            # Write user input to memory
            slot = self.memory.write(user_input)
            logger.info(f"Memory updated in slot {slot}")
            
            # Retrieve top related memory entries
            indices, values, entries = self.memory.read(user_input, top_k=3)
            print("\nTop memory entries related to your input:")
            for i, entry in enumerate(entries):
                print(f"  {i+1}. {entry} (similarity: {values[i].item():.4f})")
            
            # === NEW: Capture image and perform OCR after user response ===
            ocr_text = ""
            image = self.image.capture_image()
            if image is not None:
                ocr_text = self.image.ocr_image(image)
                print(f"\n[OCR extracted text: {ocr_text}]")
            else:
                logger.warning("No image captured for OCR")
            
            # Build conversation prompt with both text entries and OCR text
            conversation_history = "\n".join(self.memory.entries[-5:])
            prompt = (
                f"Conversation so far:\n{conversation_history}\n"
                f"User's question: '{user_input}'\n"
                f"Additional context from image: {ocr_text}\n"
                "Answer the question directly and concisely."
            )
            
            # Generate response
            if special_response:
                bot_response = special_response
            else:
                bot_response = self.llm.generate_response(prompt)
            
            # Respond to the user
            self.respond(bot_response)
            
            # Save updated memory
            self.memory.save_memory()


    def cleanup(self) -> None:
        """Clean up resources before exiting."""
        self.memory.save_memory()
        self.image.release_resources()
        logger.info("Resources cleaned up") 