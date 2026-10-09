"""
Embeddings Module
Generates embeddings using sentence-transformers.
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import time


class EmbeddingGenerator:
    """Handles embedding generation with batching for memory efficiency."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model
            batch_size: Batch size for embedding generation (reduces memory usage)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts with batching.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise Exception("Model not loaded")
        
        embeddings = []
        total_texts = len(texts)
        
        print(f"Generating embeddings for {total_texts} texts in batches of {self.batch_size}")
        
        start_time = time.time()
        
        for i in range(0, total_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            embeddings.extend(batch_embeddings)
            
            if (i + self.batch_size) % (self.batch_size * 5) == 0:
                print(f"Processed {min(i + self.batch_size, total_texts)}/{total_texts} texts")
        
        elapsed = time.time() - start_time
        print(f"Embedding generation completed in {elapsed:.2f} seconds")
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 384  # Default for all-MiniLM-L6-v2
