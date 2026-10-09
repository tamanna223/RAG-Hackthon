"""
Evaluation Module
Tests the RAG pipeline with sample queries and calculates metrics.
"""

import time
from typing import List, Dict, Set
import os
from dotenv import load_dotenv
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag import RAGPipeline
from reranking import CrossEncoderReranker, LexicalReranker

# Load environment variables
load_dotenv()


class RAGEvaluator:
    """Evaluates RAG pipeline performance."""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        rag_pipeline: RAGPipeline,
        use_reranking: bool = False
    ):
        """
        Initialize evaluator.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: VectorStore instance
            rag_pipeline: RAGPipeline instance
            use_reranking: Whether to use cross-encoder reranking
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.rag_pipeline = rag_pipeline
        self.use_reranking = use_reranking
        
        # Initialize reranker if enabled
        self.reranker = None
        if use_reranking:
            try:
                self.reranker = CrossEncoderReranker()
                if self.reranker.is_available():
                    print("Cross-encoder reranking enabled")
                else:
                    print("Cross-encoder not available, using lexical reranking")
                    self.reranker = LexicalReranker()
            except Exception as e:
                print(f"Failed to initialize reranker: {e}")
                self.reranker = LexicalReranker()
    
    def evaluate_query(self, query: str, top_k: int = 5) -> Dict[str, any]:
        """
        Evaluate a single query end-to-end.
        
        Args:
            query: Test query
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"\nEvaluating query: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_generator.model.encode([query])[0]
        
        # Retrieve chunks with optional reranking
        start_time = time.time()
        if self.use_reranking and self.reranker:
            retrieved_chunks, retrieval_time = self.vector_store.search_with_rerank(
                query_embedding, query, top_k, self.reranker
            )
        else:
            retrieved_chunks, retrieval_time = self.vector_store.search(query_embedding, top_k)
        
        # Generate answer
        result = self.rag_pipeline.generate_answer(query, retrieved_chunks)
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "answer": result["answer"],
            "citations": result["citations"],
            "retrieved_chunks": retrieved_chunks,
            "retrieval_latency": retrieval_time,
            "generation_latency": result["generation_time"],
            "total_latency": total_time,
            "chunks_retrieved": len(retrieved_chunks)
        }
    
    def run_evaluation(self, queries: List[str]) -> List[Dict[str, any]]:
        """
        Run evaluation on multiple queries.
        
        Args:
            queries: List of test queries
            
        Returns:
            List of evaluation results
        """
        results = []
        
        print(f"Running evaluation on {len(queries)} queries...")
        
        for query in queries:
            try:
                result = self.evaluate_query(query)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating query '{query}': {str(e)}")
                results.append({
                    "query": query,
                    "error": str(e)
                })
        
        return results
    
    def calculate_recall_at_k(self, results: List[Dict[str, any]], relevant_docs: Dict[str, Set[str]], k: int = 5) -> float:
        """
        Calculate Recall@K metric.
        
        Args:
            results: Evaluation results
            relevant_docs: Dictionary mapping query to set of relevant document IDs
            k: K value for Recall@K
            
        Returns:
            Recall@K score
        """
        if not relevant_docs:
            return 0.0
        
        total_recall = 0.0
        count = 0
        
        for result in results:
            if "error" in result:
                continue
            
            query = result["query"]
            if query not in relevant_docs:
                continue
            
            retrieved_doc_ids = {
                f"{chunk['filename']}_page_{chunk['page_number']}" 
                for chunk in result["retrieved_chunks"][:k]
            }
            
            relevant = relevant_docs[query]
            if relevant:
                recall = len(retrieved_doc_ids & relevant) / len(relevant)
                total_recall += recall
                count += 1
        
        return total_recall / count if count > 0 else 0.0
    
    def calculate_mrr(self, results: List[Dict[str, any]], relevant_docs: Dict[str, Set[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            results: Evaluation results
            relevant_docs: Dictionary mapping query to set of relevant document IDs
            
        Returns:
            MRR score
        """
        if not relevant_docs:
            return 0.0
        
        total_rr = 0.0
        count = 0
        
        for result in results:
            if "error" in result:
                continue
            
            query = result["query"]
            if query not in relevant_docs:
                continue
            
            relevant = relevant_docs[query]
            if not relevant:
                continue
            
            # Find first relevant document
            for i, chunk in enumerate(result["retrieved_chunks"]):
                doc_id = f"{chunk['filename']}_page_{chunk['page_number']}"
                if doc_id in relevant:
                    total_rr += 1.0 / (i + 1)
                    count += 1
                    break
        
        return total_rr / count if count > 0 else 0.0
    
    def calculate_citation_accuracy(self, results: List[Dict[str, any]]) -> float:
        """
        Calculate citation accuracy (percentage of answers with citations).
        
        Args:
            results: Evaluation results
            
        Returns:
            Citation accuracy score
        """
        if not results:
            return 0.0
        
        with_citations = sum(1 for r in results if "error" not in r and r.get("citations"))
        total = sum(1 for r in results if "error" not in r)
        
        return with_citations / total if total > 0 else 0.0
    
    def print_summary(self, results: List[Dict[str, any]], relevant_docs: Dict[str, Set[str]] = None):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        print(f"Total queries: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            avg_retrieval = sum(r["retrieval_latency"] for r in successful) / len(successful)
            avg_generation = sum(r["generation_latency"] for r in successful) / len(successful)
            avg_total = sum(r["total_latency"] for r in successful) / len(successful)
            
            print(f"\nAverage latencies:")
            print(f"  Retrieval: {avg_retrieval:.3f}s")
            print(f"  Generation: {avg_generation:.3f}s")
            print(f"  Total: {avg_total:.3f}s")
            
            # Calculate metrics if relevant docs provided
            if relevant_docs:
                recall_5 = self.calculate_recall_at_k(successful, relevant_docs, k=5)
                recall_10 = self.calculate_recall_at_k(successful, relevant_docs, k=10)
                mrr = self.calculate_mrr(successful, relevant_docs)
                
                print(f"\nRetrieval metrics:")
                print(f"  Recall@5: {recall_5:.3f}")
                print(f"  Recall@10: {recall_10:.3f}")
                print(f"  MRR: {mrr:.3f}")
            
            # Citation accuracy
            citation_acc = self.calculate_citation_accuracy(successful)
            print(f"\nCitation accuracy: {citation_acc:.3f}")
        
        if failed:
            print(f"\nFailed queries:")
            for r in failed:
                print(f"  - {r['query']}: {r['error']}")


if __name__ == "__main__":
    # Sample evaluation script
    print("Loading vector store...")
    
    vector_store = VectorStore()
    if vector_store.exists():
        vector_store.load()
    else:
        print("No existing index found. Please ingest PDFs first.")
        exit(1)
    
    print("Loading embedding model...")
    embedding_generator = EmbeddingGenerator()
    
    print("Initializing RAG pipeline...")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    rag_pipeline = RAGPipeline(ollama_url, ollama_model, groq_api_key, groq_model)
    
    # Enable reranking
    use_reranking = os.getenv("USE_RERANKING", "false").lower() == "true"
    evaluator = RAGEvaluator(embedding_generator, vector_store, rag_pipeline, use_reranking)
    
    # Sample queries (customize based on your documents)
    sample_queries = [
        "What is machine learning?",
        "Explain neural networks.",
        "What is deep learning?"
    ]
    
    results = evaluator.run_evaluation(sample_queries)
    evaluator.print_summary(results)
