"""
Reranking Module
Uses cross-encoder to rerank retrieved chunks for better relevance.
"""

from typing import List, Dict, Optional
import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Reranks retrieved chunks using cross-encoder."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Name of the cross-encoder model
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load the cross-encoder model."""
        if self.model is None:
            print(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name, device=self.device)
            print("Cross-encoder model loaded successfully")
    
    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Rerank retrieved chunks based on query relevance.
        
        Args:
            query: User's query
            retrieved_chunks: List of retrieved chunks with metadata
            top_k: Number of top results to return (default: all)
            
        Returns:
            Reranked list of chunks
        """
        if not retrieved_chunks:
            return retrieved_chunks
        
        # Load model if not loaded
        self.load_model()
        
        # Prepare query-chunk pairs
        pairs = [[query, chunk['text']] for chunk in retrieved_chunks]
        
        # Compute cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Add scores to chunks
        for chunk, score in zip(retrieved_chunks, scores):
            chunk['cross_encoder_score'] = float(score)
        
        # Sort by cross-encoder score (descending)
        reranked = sorted(retrieved_chunks, key=lambda x: x['cross_encoder_score'], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked
    
    def is_available(self) -> bool:
        """Check if cross-encoder is available."""
        try:
            self.load_model()
            return True
        except Exception as e:
            print(f"Cross-encoder not available: {str(e)}")
            return False


class LexicalReranker:
    """Simple lexical reranker based on keyword matching."""
    
    def __init__(self):
        """Initialize lexical reranker."""
        pass
    
    def extract_keywords(self, text: str) -> set:
        """Extract keywords from text."""
        # Simple keyword extraction: lowercase words, remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                     'during', 'before', 'after', 'above', 'below', 'between', 'under',
                     'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                     'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                     'very', 'just', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                     'while', 'about', 'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'out',
                     'off', 'over', 'under', 'again', 'further', 'then', 'once'}
        
        words = text.lower().split()
        keywords = {word for word in words if word.isalnum() and word not in stopwords}
        return keywords
    
    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Rerank chunks based on keyword overlap.
        
        Args:
            query: User's query
            retrieved_chunks: List of retrieved chunks
            top_k: Number of top results to return
            
        Returns:
            Reranked list of chunks
        """
        if not retrieved_chunks:
            return retrieved_chunks
        
        # Extract query keywords
        query_keywords = self.extract_keywords(query)
        
        # Calculate keyword overlap for each chunk
        for chunk in retrieved_chunks:
            chunk_keywords = self.extract_keywords(chunk['text'])
            overlap = len(query_keywords & chunk_keywords)
            chunk['lexical_score'] = overlap
        
        # Sort by lexical score (descending)
        reranked = sorted(retrieved_chunks, key=lambda x: x['lexical_score'], reverse=True)
        
        # Return top-k if specified
        if top_k is not None:
            reranked = reranked[:top_k]
        
        return reranked
