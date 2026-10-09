"""
PDF Ingestion Module
Extracts text from PDFs page by page, preserving filename and page numbers.
"""

import pypdfium2  # Alternative to PyMuPDF with pre-built wheels
from typing import List, Dict, Optional
import os
from preprocessing import TextPreprocessor


class PDFIngestion:
    """Handles PDF text extraction with page-level granularity."""
    
    def __init__(self, use_preprocessing: bool = True):
        self.documents = []
        self.preprocessor = TextPreprocessor() if use_preprocessing else None
        self.use_preprocessing = use_preprocessing
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from a single PDF page by page.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing filename, page_number, and text
        """
        pages = []
        filename = os.path.basename(pdf_path)
        
        try:
            doc = pypdfium2.PdfDocument(pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_textpage().get_text_range()
                
                if text.strip():  # Only add non-empty pages
                    # Apply preprocessing if enabled
                    if self.use_preprocessing and self.preprocessor:
                        preprocessed = self.preprocessor.preprocess_page(text, page_num + 1)
                        text_to_use = preprocessed['cleaned_text']
                        language = preprocessed['language']
                    else:
                        text_to_use = text.strip()
                        language = 'en'  # Default
                    
                    # Only add if text is valid
                    if self.preprocessor and self.preprocessor.is_valid_text(text_to_use):
                        pages.append({
                            "filename": filename,
                            "page_number": page_num + 1,  # 1-based page numbers
                            "text": text_to_use,
                            "language": language
                        })
                    elif not self.preprocessor and text.strip():
                        # If no preprocessor, add anyway
                        pages.append({
                            "filename": filename,
                            "page_number": page_num + 1,
                            "text": text.strip(),
                            "language": 'en'
                        })
            
            doc.close()
            return pages
            
        except Exception as e:
            raise Exception(f"Error extracting text from {filename}: {str(e)}")
    
    def ingest_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict[str, any]]:
        """
        Extract text from multiple PDFs.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of all pages from all PDFs
        """
        all_pages = []
        
        for pdf_path in pdf_paths:
            try:
                pages = self.extract_text_from_pdf(pdf_path)
                all_pages.extend(pages)
                print(f"Extracted {len(pages)} pages from {os.path.basename(pdf_path)}")
            except Exception as e:
                print(f"Failed to process {pdf_path}: {str(e)}")
        
        self.documents = all_pages
        return all_pages
    
    def get_total_pages(self) -> int:
        """Return total number of pages extracted."""
        return len(self.documents)
