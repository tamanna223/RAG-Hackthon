"""
Preprocessing Module
Handles text cleaning, header/footer removal, and language detection.
"""

import re
from typing import List, Tuple
from langdetect import detect, LangDetectException


class TextPreprocessor:
    """Handles text preprocessing for RAG pipeline."""
    
    def __init__(self):
        """Initialize preprocessor."""
        # Common header/footer patterns
        self.header_patterns = [
            r'^\d+\s*[A-Z][a-z]+\s+\d{4}',  # Page number + month + year
            r'^Chapter\s+\d+',  # Chapter headers
            r'^Section\s+\d+',  # Section headers
            r'^\d+\s*$'  # Just page numbers
        ]
        
        self.footer_patterns = [
            r'\d+$',  # Page numbers at end
            r'^Page\s+\d+\s+of\s+\d+',  # Page X of Y
            r'^©\s*\d{4}',  # Copyright
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,;:!?\'"-]', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def remove_headers_footers(self, text: str, lines: List[str]) -> Tuple[str, List[str]]:
        """
        Remove headers and footers from text.
        
        Args:
            text: Full text
            lines: List of lines from the page
            
        Returns:
            Tuple of (cleaned text, filtered lines)
        """
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches header/footer patterns
            is_header = any(re.match(pattern, line) for pattern in self.header_patterns)
            is_footer = any(re.search(pattern, line) for pattern in self.footer_patterns)
            
            # Skip if it's a header or footer
            if not (is_header or is_footer):
                filtered_lines.append(line)
        
        cleaned_text = ' '.join(filtered_lines)
        return cleaned_text, filtered_lines
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr')
        """
        try:
            # Use only first 1000 chars for faster detection
            return detect(text[:1000])
        except LangDetectException:
            # Default to English if detection fails
            return 'en'
    
    def preprocess_page(self, text: str, page_num: int = None) -> dict:
        """
        Preprocess a page of text.
        
        Args:
            text: Raw text from page
            page_num: Page number (optional)
            
        Returns:
            Dictionary with preprocessed text and metadata
        """
        lines = text.split('\n')
        
        # Remove headers and footers
        cleaned_text, filtered_lines = self.remove_headers_footers(text, lines)
        
        # Clean text
        cleaned_text = self.clean_text(cleaned_text)
        
        # Detect language
        language = self.detect_language(cleaned_text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'filtered_lines': filtered_lines,
            'language': language,
            'page_num': page_num
        }
    
    def is_valid_text(self, text: str, min_length: int = 50) -> bool:
        """
        Check if text is valid (not empty, not just noise).
        
        Args:
            text: Text to validate
            min_length: Minimum character length
            
        Returns:
            True if text is valid
        """
        if not text or len(text.strip()) < min_length:
            return False
        
        # Check if text contains meaningful content (not just symbols)
        words = text.split()
        if len(words) < 5:
            return False
        
        return True
