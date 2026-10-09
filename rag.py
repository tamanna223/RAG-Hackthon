"""
RAG Pipeline Module
Orchestrates retrieval and answer generation.
"""

import requests
import time
from typing import List, Dict, Optional
import os
from groq import Groq


class RAGPipeline:
    """Handles the complete RAG pipeline: retrieval and generation."""
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3.2:1b",
        groq_api_key: Optional[str] = None,
        groq_model: str = "llama3-8b-8192"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            ollama_base_url: Base URL for Ollama API
            ollama_model: Model name to use in Ollama
            groq_api_key: API key for Groq (optional, faster)
            groq_model: Model name for Groq
        """
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "llama3-8b-8192")
        
        # Initialize Groq client if API key is available
        self.use_groq = bool(self.groq_api_key and self.groq_api_key != "your_groq_api_key_here")
        if self.use_groq:
            self.groq_client = Groq(api_key=self.groq_api_key)
            print(f"Using Groq API with model: {self.groq_model}")
        else:
            print("Groq API key not configured, using Ollama as fallback")
    
    def check_ollama_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Ollama not accessible: {str(e)}")
            return False
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, any]]
    ) -> Dict[str, any]:
        """
        Generate answer using retrieved context.
        
        Args:
            query: User's question
            retrieved_chunks: List of retrieved chunks with metadata
            
        Returns:
            Dictionary with answer, citations, and timing information
        """
        start_time = time.time()
        
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(
                f"[{chunk['filename']}, page {chunk['page_number']}]: {chunk['text']}"
            )
        context = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = f"""Answer using ONLY this context:
{context}

Question: {query}

Answer:"""
        
        try:
            # Use Groq if available (faster)
            if self.use_groq:
                response = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Answer questions using ONLY the provided context. Cite sources as [filename, page N]."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=200,
                    timeout=10
                )
                answer = response.choices[0].message.content.strip()
            else:
                # Fallback to Ollama
                response = requests.post(
                    f"{self.ollama_base_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 150,
                            "num_ctx": 2048
                        }
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise Exception(f"Ollama API error: {response.status_code}")
                
                result = response.json()
                answer = result.get("response", "").strip()
            
            generation_time = time.time() - start_time
            
            # Extract unique citations
            citations = list(set([
                f"{chunk['filename']}, page {chunk['page_number']}"
                for chunk in retrieved_chunks
            ]))
            
            return {
                "answer": answer,
                "citations": citations,
                "context_used": len(retrieved_chunks),
                "generation_time": generation_time,
                "model_used": "Groq" if self.use_groq else "Ollama"
            }
            
        except requests.exceptions.Timeout:
            raise Exception("Request timed out.")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to API.")
        except Exception as e:
            raise Exception(f"Error generating answer: {str(e)}")
