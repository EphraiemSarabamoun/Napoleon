import os
import json
import subprocess
import hashlib
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from src.utils.config import logger, MODEL_NAME

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
        """
        Create a deterministic embedding using SHA-256 as a fallback.
        """
        logger.info("Using fallback embedding method")
        h = hashlib.sha256(text.encode('utf-8')).digest()
        v = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
        
        if len(v) < self.embedding_dim:
            v = np.pad(v, (0, self.embedding_dim - len(v)), mode='constant')
        else:
            v = v[:self.embedding_dim]
            
        return torch.tensor(v, dtype=torch.float)
    
    def write(self, text: str) -> int:
        """
        Stores a new memory entry by generating its embedding.
        Chooses the memory slot with the lowest usage.
        """
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
        """
        Retrieve memory entries similar to the query text.
        """
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
        """Save the current memory state to disk."""
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
        """Load the memory state from disk."""
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
        """
        Create an embedding for the given text using the LLM.
        Falls back to a simple hash-based embedding if the model fails.
        """
        try:
            command = ["ollama", "run", MODEL_NAME, text]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                encoding="utf-8",
                errors="replace",
                timeout=60  # Increased timeout from 30 to 60 seconds
            )
            output_text = result.stdout.strip()
            if not output_text:
                logger.warning(f"No output received from {MODEL_NAME}")
                return self._fallback_embedding(text)
                    
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
                    logger.warning("Response doesn't contain embedding")
                    response_text = output_json.get("response", output_text)
                    return self._fallback_embedding(response_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response")
                return self._fallback_embedding(output_text)
                    
        except subprocess.CalledProcessError as e:
            logger.exception("Error obtaining embedding")
            return self._fallback_embedding(text)
        except subprocess.TimeoutExpired:
            logger.exception("Embedding generation timed out")
            return self._fallback_embedding(text)
        except Exception as e:
            logger.exception("Unexpected error obtaining embedding")
            return self._fallback_embedding(text) 