"""
Vector Store Module
Manages FAISS index for similarity search.
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
import time


class VectorStore:
    """Handles FAISS index creation, persistence, and similarity search."""
    
    def __init__(self, index_path: str = "faiss_index", metadata_path: str = "chunk_metadata.pkl"):
        """
        Initialize vector store.
        
        Args:
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load chunk metadata
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.chunk_metadata = []
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
    
    def create_index(self, embeddings: List[np.ndarray], metadata: List[Dict[str, any]]):
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: List of embedding vectors
            metadata: List of chunk metadata corresponding to embeddings
        """
        if len(embeddings) == 0:
            raise Exception("No embeddings provided")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Get embedding dimension
        self.embedding_dim = embeddings_array.shape[1]
        
        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings_array)
        
        # Store metadata
        self.chunk_metadata = metadata
        
        print(f"Created FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Embedding of the query
            top_k: Number of results to return
            
        Returns:
            List of results with metadata and similarity scores
        """
        if self.index is None:
            raise Exception("Index not created. Load or create an index first.")
        
        # Reshape query for FAISS
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, top_k)
        retrieval_time = time.time() - start_time
        
        # Prepare results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(self.chunk_metadata):
                result = self.chunk_metadata[idx].copy()
                result["similarity_score"] = float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                result["distance"] = float(distances[0][i])
                results.append(result)
        
        return results, retrieval_time
    
    def search_with_rerank(
        self,
        query_embedding: np.ndarray,
        query: str,
        top_k: int = 5,
        reranker: Optional[object] = None
    ) -> Tuple[List[Dict[str, any]], float]:
        """
        Search for similar chunks with optional reranking.
        
        Args:
            query_embedding: Embedding of the query
            query: Original query text (for reranking)
            top_k: Number of results to return
            reranker: Optional reranker object
            
        Returns:
            Tuple of (results, retrieval_time)
        """
        # First, do initial search
        initial_top_k = top_k * 2 if reranker else top_k  # Get more for reranking
        results, retrieval_time = self.search(query_embedding, initial_top_k)
        
        # Apply reranking if provided
        if reranker:
            results = reranker.rerank(query, results, top_k)
        
        return results, retrieval_time
    
    def save(self):
        """Save index and metadata to disk."""
        if self.index is None:
            raise Exception("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.chunk_metadata, f)
        
        print(f"Saved index to {self.index_path} and metadata to {self.metadata_path}")
    
    def load(self):
        """Load index and metadata from disk."""
        if not os.path.exists(self.index_path):
            raise Exception(f"Index file not found: {self.index_path}")
        
        if not os.path.exists(self.metadata_path):
            raise Exception(f"Metadata file not found: {self.metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        
        # Load metadata
        with open(self.metadata_path, 'rb') as f:
            self.chunk_metadata = pickle.load(f)
        
        self.embedding_dim = self.index.d
        
        print(f"Loaded index with {self.index.ntotal} vectors from {self.index_path}")
    
    def exists(self) -> bool:
        """Check if saved index and metadata exist."""
        return os.path.exists(self.index_path) and os.path.exists(self.metadata_path)
    
    def get_chunk_count(self) -> int:
        """Return total number of chunks in the index."""
        if self.index:
            return self.index.ntotal
        return 0
