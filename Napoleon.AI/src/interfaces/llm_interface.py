import json
import subprocess

from src.utils.config import logger, MODEL_NAME

class LLMInterface:
    """Handles interactions with the LLM."""
    
    @staticmethod
    def generate_response(prompt: str, model_name: str = MODEL_NAME, timeout: int = 60) -> str:
        """
        Calls the LLM to generate a response based on the prompt.
        """
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
                timeout=timeout
            )
            
            output_text = result.stdout.strip()
            if not output_text:
                logger.warning(f"No output received from {model_name}")
                return "[No response generated]"
            
            try:
                output_json = json.loads(output_text)
                response = output_json.get("response", "")
                if not response:
                    response = output_text
            except json.JSONDecodeError:
                response = output_text
                
            return response
            
        except subprocess.CalledProcessError as e:
            logger.exception("Error generating response")
            return "[Error generating response]"
        except subprocess.TimeoutExpired:
            logger.exception("Response generation timed out")
            return "[Response generation timed out]"
        except Exception as e:
            logger.exception("Unexpected error generating response")
            return "[Error generating response]" 