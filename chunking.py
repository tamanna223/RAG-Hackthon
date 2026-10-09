"""
Text Chunking Module
Creates deterministic overlapping chunks from extracted text.
"""

from typing import List, Dict
import re


class TextChunker:
    """Handles text chunking with overlap for better context retention."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        """
        Initialize chunker with size and overlap.
        
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Split documents into overlapping chunks.
        
        Args:
            documents: List of document pages with filename, page_number, and text
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for doc in documents:
            text = doc["text"]
            filename = doc["filename"]
            page_number = doc["page_number"]
            
            # Split text into chunks with overlap
            start = 0
            chunk_id = 0
            
            while start < len(text):
                end = start + self.chunk_size
                
                # Try to break at word boundary
                if end < len(text):
                    # Find last space before chunk_size
                    last_space = text.rfind(' ', start, end)
                    if last_space != -1:
                        end = last_space + 1
                
                chunk_text = text[start:end].strip()
                
                if chunk_text:  # Only add non-empty chunks
                    chunks.append({
                        "chunk_id": chunk_id,
                        "filename": filename,
                        "page_number": page_number,
                        "text": chunk_text,
                        "char_start": start,
                        "char_end": end
                    })
                    chunk_id += 1
                
                # Move start position with overlap
                start = end - self.chunk_overlap
                if start < 0:
                    start = 0
        
        return chunks
    
    def get_chunk_count(self, chunks: List[Dict[str, any]]) -> int:
        """Return total number of chunks."""
        return len(chunks)
