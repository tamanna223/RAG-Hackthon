"""
Streamlit App for RAG PDF Chatbot
Main application interface.
"""

import streamlit as st
import os
import time
from dotenv import load_dotenv
from ingestion import PDFIngestion
from chunking import TextChunker
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from rag import RAGPipeline

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

# Custom CSS - Modern Chat UI
st.markdown("""
<style>
    /* Global styles */
    .main {
        background-color: #1a1a2e !important;
    }
    
    .main .block-container {
        background-color: #1a1a2e !important;
        max-width: 1200px !important;
        padding: 2rem 1rem !important;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #a0aec0;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling - Modern bubbles */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        margin: 1rem 0 !important;
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        display: flex;
        justify-content: flex-end;
    }
    
    .stChatMessage[data-testid="user-message"] > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 20px 20px 5px 20px !important;
        padding: 1rem 1.5rem !important;
        max-width: 70% !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        display: flex;
        justify-content: flex-start;
    }
    
    .stChatMessage[data-testid="assistant-message"] > div {
        background-color: #2d3748 !important;
        color: #e2e8f0 !important;
        border-radius: 20px 20px 20px 5px !important;
        padding: 1rem 1.5rem !important;
        max-width: 70% !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* All text in messages */
    .stChatMessage p,
    .stChatMessage div,
    .stChatMessage span,
    .stChatMessage strong {
        color: inherit !important;
    }
    
    /* Citation styling */
    .citation-box {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 4px solid #667eea;
        font-size: 0.9rem;
        color: #e2e8f0 !important;
    }
    
    /* Passage styling */
    .passage-box {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 3px solid #48bb78;
        color: #e2e8f0 !important;
    }
    
    /* Metric card styling */
    div[data-testid="stMetricValue"] {
        background-color: #2d3748 !important;
        padding: 0.75rem;
        border-radius: 12px;
        border: 1px solid #4a5568;
        color: #e2e8f0 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #48bb78 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #667eea !important;
        border-radius: 16px;
        padding: 2rem;
        background-color: #2d3748 !important;
    }
    
    .stFileUploader label,
    .stFileUploader span,
    .stFileUploader div,
    .stFileUploader p {
        color: #e2e8f0 !important;
    }
    
    .stFileUploader button {
        background-color: #667eea !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #2d3748;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        color: #a0aec0;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .css-1d391kg h3,
    .css-1d391kg p,
    .css-1d391kg span,
    .css-1d391kg label,
    .css-1d391kg div {
        color: white !important;
    }
    
    /* Chat input styling */
    .stChatInput {
        background-color: #2d3748 !important;
        border: 2px solid #667eea !important;
        border-radius: 24px !important;
    }
    
    .stChatInput input,
    .stChatInput textarea {
        color: #e2e8f0 !important;
        background-color: transparent !important;
    }
    
    .stChatInput input::placeholder {
        color: #718096 !important;
    }
    
    /* Text input styling */
    .stTextInput input {
        color: #e2e8f0 !important;
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
        border-radius: 8px;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        color: #e2e8f0 !important;
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
        border-radius: 8px;
    }
    
    /* ALL input elements */
    input[type="text"],
    input[type="email"],
    input[type="password"],
    textarea {
        color: #e2e8f0 !important;
        background-color: #2d3748 !important;
        border: 1px solid #4a5568 !important;
    }
    
    /* Warning message styling */
    .stAlert {
        background-color: #2d3748 !important;
        color: #e2e8f0 !important;
        border: 1px solid #667eea !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2d3748 !important;
        border-radius: 12px;
        font-weight: 600;
        color: #e2e8f0 !important;
    }
    
    /* All main content text */
    .main p,
    .main div:not(.stChatMessage),
    .main span,
    .main h1,
    .main h2,
    .main h3 {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)


def check_dependencies():
    """Check if required dependencies are available."""
    errors = []
    
    # Check Ollama
    try:
        import requests
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{ollama_url}/api/tags", timeout=2)
        if response.status_code != 200:
            errors.append("Ollama is not running or not accessible")
    except:
        errors.append("Ollama is not running or not accessible")
    
    # Check Tesseract (optional)
    try:
        import pytesseract
        tesseract_cmd = os.getenv("TESSERACT_CMD")
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pass  # Tesseract is available
        else:
            st.warning("Tesseract OCR not configured. Scanned PDFs may not be processed correctly.")
    except:
        st.warning("Tesseract OCR not available. Scanned PDFs may not be processed correctly.")
    
    return errors


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if 'embedding_generator' not in st.session_state:
        st.session_state.embedding_generator = None
    
    if 'rag_pipeline' not in st.session_state:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
        groq_api_key = os.getenv("GROQ_API_KEY")
        groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
        st.session_state.rag_pipeline = RAGPipeline(ollama_url, ollama_model, groq_api_key, groq_model)
    
    if 'index_loaded' not in st.session_state:
        st.session_state.index_loaded = False
    
    if 'ingestion_complete' not in st.session_state:
        st.session_state.ingestion_complete = False


def main():
    """Main application logic."""
    
    # Header
    st.markdown('<div class="main-header">📚 RAG PDF Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload PDFs and ask questions about their content</div>', unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Check dependencies
    errors = check_dependencies()
    if errors:
        st.error("Dependency Errors:")
        for error in errors:
            st.error(f"  - {error}")
        st.error("Please resolve these errors before proceeding.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        
        # Display chunk size and overlap from env
        chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "128"))
        top_k = int(os.getenv("TOP_K", "5"))
        
        st.markdown(f"**📏 Chunk Size:** <span style='color: white; font-weight: bold; font-size: 1.2rem;'>{chunk_size}</span>", unsafe_allow_html=True)
        st.markdown(f"**🔄 Chunk Overlap:** <span style='color: white; font-weight: bold; font-size: 1.2rem;'>{chunk_overlap}</span>", unsafe_allow_html=True)
        st.markdown(f"**🎯 Top-K Results:** <span style='color: white; font-weight: bold; font-size: 1.2rem;'>{top_k}</span>", unsafe_allow_html=True)
        
        st.markdown("### 📊 Index Status")
        st.markdown("---")
        if st.session_state.vector_store.exists():
            st.success(f"✅ Index exists with {st.session_state.vector_store.get_chunk_count()} chunks")
            if st.button("📂 Load Existing Index", use_container_width=True):
                try:
                    st.session_state.vector_store.load()
                    st.session_state.index_loaded = True
                    st.success("✅ Index loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading index: {str(e)}")
        else:
            st.info("ℹ️ No existing index found")
        
        st.markdown("### 💻 System Info")
        st.markdown("---")
        
        # Show LLM model being used
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "your_groq_api_key_here":
            st.markdown(f"**🤖 LLM Model:** `{os.getenv('GROQ_MODEL', 'llama3-8b-8192')}` (Groq API)")
        else:
            st.markdown(f"**🤖 LLM Model:** `{os.getenv('OLLAMA_MODEL', 'llama3.2:1b')}` (Ollama Fallback)")
        
        st.markdown(f"**🧠 Embedding Model:** `all-MiniLM-L6-v2`")
        st.markdown(f"**📦 Vector Store:** `FAISS CPU`")
        
        st.markdown("---")
        st.markdown("### 📖 Quick Guide")
        st.markdown("""
        1. **Upload PDFs** in the Ingest tab
        2. **Process** to create the index
        3. **Ask questions** in the Chat tab
        4. **View citations** and passages
        """)
    
    # Main content area
    tab1, tab2 = st.tabs(["Ingest PDFs", "Chat"])
    
    # Tab 1: PDF Ingestion
    with tab1:
        st.markdown("### 📄 Upload PDFs")
        st.markdown("Upload your PDF documents to create a searchable knowledge base.")
        
        uploaded_files = st.file_uploader(
            "📁 Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            label_visibility="visible"
        )
        
        if uploaded_files:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📚 Files Selected", len(uploaded_files))
            with col2:
                total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
                st.metric("💾 Total Size", f"{total_size:.2f} MB")
            with col3:
                st.metric("📊 Status", "Ready to Process")
            
            st.markdown("---")
            
            if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing PDFs..."):
                    try:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Save uploaded files temporarily
                        temp_dir = "temp_pdfs"
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        pdf_paths = []
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            pdf_paths.append(file_path)
                        
                        progress_bar.progress(20)
                        status_text.text("📖 Extracting text from PDFs...")
                        
                        # Extract text
                        ingestion = PDFIngestion()
                        documents = ingestion.ingest_multiple_pdfs(pdf_paths)
                        
                        progress_bar.progress(40)
                        status_text.text(f"✅ Extracted {len(documents)} pages")
                        
                        # Chunk text
                        status_text.text("✂️ Creating chunks...")
                        chunker = TextChunker(chunk_size, chunk_overlap)
                        chunks = chunker.chunk_text(documents)
                        
                        progress_bar.progress(60)
                        status_text.text(f"✅ Created {len(chunks)} chunks")
                        
                        # Generate embeddings
                        status_text.text("🧠 Generating embeddings (this may take a while)...")
                        if st.session_state.embedding_generator is None:
                            st.session_state.embedding_generator = EmbeddingGenerator()
                        
                        texts = [chunk["text"] for chunk in chunks]
                        embeddings = st.session_state.embedding_generator.generate_embeddings(texts)
                        
                        progress_bar.progress(80)
                        status_text.text("🔨 Building vector index...")
                        
                        # Create and save index
                        st.session_state.vector_store.create_index(embeddings, chunks)
                        st.session_state.vector_store.save()
                        
                        progress_bar.progress(100)
                        
                        st.session_state.index_loaded = True
                        st.session_state.ingestion_complete = True
                        
                        # Cleanup
                        for pdf_path in pdf_paths:
                            os.remove(pdf_path)
                        os.rmdir(temp_dir)
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Success message with stats
                        st.markdown("### ✅ Processing Complete!")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("📄 PDFs Processed", len(uploaded_files))
                        with col2:
                            st.metric("📑 Pages Extracted", len(documents))
                        with col3:
                            st.metric("🧩 Chunks Created", len(chunks))
                        with col4:
                            st.metric("📊 Index Size", f"{len(chunks)} vectors")
                        
                        st.success("🎉 Your knowledge base is ready! Switch to the Chat tab to ask questions.")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing PDFs: {str(e)}")
                        st.error("Please check the error message and try again.")
    
    # Tab 2: Chat
    with tab2:
        st.markdown("### 💬 Ask Questions")
        st.markdown("Ask questions about your uploaded PDFs and get answers with citations.")
        
        if not st.session_state.index_loaded:
            st.warning("⚠️ Please ingest PDFs first in the 'Ingest PDFs' tab.")
            st.info("💡 Tip: Upload some PDFs and process them to create a searchable knowledge base.")
            return
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history in chronological order (oldest first, newest last)
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])
                
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message["metadata"]
                    
                    # Show citations with better styling
                    if metadata.get("citations"):
                        st.markdown("#### 📚 Citations")
                        for citation in metadata["citations"]:
                            st.markdown(f"- 📄 {citation}")
                    
                    # Show retrieved passages
                    if metadata.get("retrieved_chunks"):
                        with st.expander("🔍 View Retrieved Passages"):
                            for i, chunk in enumerate(metadata["retrieved_chunks"]):
                                st.markdown(f"**Passage {i+1}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"📊 Similarity: `{chunk['similarity_score']:.3f}`")
                                with col2:
                                    st.markdown(f"📍 Source: `{chunk['filename']}`, page `{chunk['page_number']}`")
                                st.markdown(f"> {chunk['text']}")
                                st.markdown("---")
                    
                    # Show metrics with icons
                    st.markdown("#### ⏱️ Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🔎 Retrieval", f"{metadata['retrieval_latency']:.3f}s")
                    with col2:
                        st.metric("🤖 Generation", f"{metadata['generation_latency']:.3f}s")
                    with col3:
                        st.metric("⚡ Total", f"{metadata['total_latency']:.3f}s")
        
        # Chat input at the bottom (always visible)
        prompt = st.chat_input("💬 Ask a question about your PDFs...")
        
        # Process new input
        if prompt:
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response (don't display immediately, just add to session state)
            with st.spinner("🔍 Retrieving and generating answer..."):
                try:
                    # Generate query embedding
                    if st.session_state.embedding_generator is None:
                        st.session_state.embedding_generator = EmbeddingGenerator()
                    
                    query_embedding = st.session_state.embedding_generator.model.encode([prompt])[0]
                    
                    # Retrieve chunks
                    retrieved_chunks, retrieval_time = st.session_state.vector_store.search(
                        query_embedding,
                        top_k=top_k
                    )
                    
                    # Generate answer
                    result = st.session_state.rag_pipeline.generate_answer(prompt, retrieved_chunks)
                    
                    total_time = retrieval_time + result["generation_time"]
                    
                    # Prepare metadata
                    metadata = {
                        "citations": result["citations"],
                        "retrieved_chunks": retrieved_chunks,
                        "retrieval_latency": retrieval_time,
                        "generation_latency": result["generation_time"],
                        "total_latency": total_time
                    }
                    
                    # Add assistant response to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "metadata": metadata
                    })
                    
                    # Rerun to display the complete conversation from session state
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error("Please check that Ollama is running and try again.")


if __name__ == "__main__":
    main()
