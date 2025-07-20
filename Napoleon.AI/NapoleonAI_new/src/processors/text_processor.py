import re
from typing import List, Optional

class TextProcessor:
    """Handles text processing and name extraction."""
    
    @staticmethod
    def clean_response(response_text: str) -> str:
        """
        Remove chain-of-thought text enclosed in <think>...</think> tags.
        """
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)
        return cleaned.strip()
    
    @staticmethod
    def extract_user_name(text: str) -> Optional[str]:
        """
        Extracts a user name from text. Now supports multi-word names.
        """
        pattern = re.compile(r"my name is\s+([\w\s]+)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        return None
    
    @staticmethod
    def get_user_name_from_entries(entries: List[str], bot_name: str) -> Optional[str]:
        """
        Scans memory entries in reverse order and returns the most recent name
        that is not the bot's name.
        """
        for entry in reversed(entries):
            name = TextProcessor.extract_user_name(entry)
            if name and name.lower() != bot_name.lower():
                return name
        return None 