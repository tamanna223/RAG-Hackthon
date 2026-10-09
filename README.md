# RAG PDF Chatbot

A complete Retrieval-Augmented Generation (RAG) chatbot for answering questions from PDF documents using open-source models.

## Features

- **Multi-PDF Upload**: Upload multiple PDFs for ingestion
- **Page-level Text Extraction**: Extract text page by page with filename and page number tracking
- **Deterministic Chunking**: Overlapping chunks with configurable size and overlap
- **FAISS Vector Store**: Fast similarity search with ANN index
- **Sentence-Transformers Embeddings**: Open-source embeddings (all-MiniLM-L6-v2)
- **Groq API Integration**: Ultra-fast LLM for answer generation (sub-second latency)
- **Ollama Fallback**: Local LLM support as backup
- **Modern Chat UI**: Professional dark-themed Streamlit interface with chat bubbles
- **Retrieval Visualization**: Shows top retrieved chunks with similarity scores
- **Citation Support**: Answers include PDF filename and page number references
- **Latency Metrics**: Displays retrieval and generation times
- **Preprocessing**: Header/footer removal, language detection, text cleaning
- **Reranking**: Optional cross-encoder and lexical reranking for better relevance
- **OCR Support**: Tesseract integration for scanned PDFs
- **Evaluation Metrics**: Recall@K, MRR, citation accuracy

## Requirements

- Python 3.13+
- Windows OS
- 8 GB RAM minimum
- Groq API key (free, recommended) or Ollama (local)
- Tesseract OCR (optional, for scanned PDFs)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-hackathon
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Get Groq API key (recommended for speed):
   - Go to https://console.groq.com/
   - Sign up for free account
   - Get API key from API Keys section

5. (Optional) Install Ollama for local fallback:
   - Download from https://ollama.com/
   - Pull a model: `ollama pull qwen2.5:0.5b`

6. (Optional) Install Tesseract OCR:
   - Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Install to `C:\Program Files\Tesseract-OCR\`

## Configuration

Create a `.env` file in the project root:

```env
# Ollama Configuration (fallback)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:0.5b

# Groq Configuration (primary - faster, free)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b

# Tesseract OCR Path (Windows)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Chunking Configuration
CHUNK_SIZE=512
CHUNK_OVERLAP=128

# Retrieval Configuration
TOP_K=5

# Reranking Configuration
USE_RERANKING=false
```

## Usage

### Start the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using the Chat Interface

1. **Ingest PDFs**:
   - Go to the "Ingest PDFs" tab
   - Upload one or more PDF files
   - Click "Ingest PDFs" to process them
   - Wait for ingestion to complete

2. **Ask Questions**:
   - Go to the "Chat" tab
   - Type your question in the chat input
   - The system will:
     - Retrieve relevant chunks from the PDFs
     - Generate an answer using the LLM
     - Display citations (PDF name + page number)
     - Show retrieval visualization with similarity scores
     - Display latency metrics

### Running Evaluation

```bash
python evaluation.py
```

This will run sample queries and display:
- Retrieval latency
- Generation latency
- Total latency
- Citation accuracy
- Recall@K (if relevant docs provided)
- MRR (if relevant docs provided)

### Testing OCR

```bash
python test_ocr.py
```

This will test OCR functionality with a sample PDF.

## Project Structure

```
rag-hackathon/
├── app.py                 # Streamlit application
├── ingestion.py           # PDF text extraction with preprocessing
├── chunking.py            # Text chunking with overlap
├── embeddings.py          # Embedding generation
├── vector_store.py        # FAISS index management with reranking
├── rag.py                 # RAG pipeline orchestration (Groq + Ollama)
├── preprocessing.py       # Text preprocessing (header/footer, language detection)
├── reranking.py           # Cross-encoder and lexical reranking
├── evaluation.py          # Evaluation script with metrics
├── test_ocr.py            # OCR testing script
├── requirements.txt       # Python dependencies
├── .env                   # Configuration (create from .env.example)
├── .env.example           # Example configuration
├── README.md              # This file
├── CHALLENGE_STATUS.md    # Challenge completion status
├── GROQ_SETUP.md          # Groq API setup instructions
├── faiss_index            # FAISS index (generated)
└── chunk_metadata.pkl     # Chunk metadata (generated)
```

## Core Components

### 1. Ingestion (`ingestion.py`)
- Extracts text from PDFs page by page
- Preserves filename and page numbers
- Supports native text extraction
- Optional OCR for scanned pages
- Integrated preprocessing (header/footer removal, language detection)

### 2. Chunking (`chunking.py`)
- Splits text into overlapping chunks
- Configurable chunk size and overlap
- Preserves metadata (filename, page number)

### 3. Embeddings (`embeddings.py`)
- Uses sentence-transformers (all-MiniLM-L6-v2)
- Generates embeddings for chunks
- Batches embeddings for efficiency

### 4. Vector Store (`vector_store.py`)
- Uses FAISS for similarity search
- Persists index and metadata
- Fast ANN search with L2 distance
- Optional reranking support

### 5. RAG Pipeline (`rag.py`)
- Orchestrates retrieval and generation
- Uses Groq API for ultra-fast generation (primary)
- Uses Ollama as fallback
- Includes citations in responses
- Measures latency

### 6. Preprocessing (`preprocessing.py`)
- Header/footer removal
- Language detection
- Text cleaning and normalization
- Validation of text quality

### 7. Reranking (`reranking.py`)
- Cross-encoder reranking (MS MARCO MiniLM)
- Lexical reranking (keyword overlap)
- Improves retrieval relevance

### 8. Evaluation (`evaluation.py`)
- Tests RAG pipeline with sample queries
- Measures retrieval and generation latency
- Calculates Recall@K, MRR, citation accuracy
- Reports success/failure rates

## Performance

- **Retrieval Latency**: ~0.01s
- **Generation Latency**: ~0.4s (with Groq API)
- **Total Latency**: ~0.4s (well within 2-5s requirement)

## Challenge Completion

This project meets all requirements for the RAG Chatbot challenge:

✅ **Dataset**: 12 PDFs (200-285 pages each)
✅ **Core AI Tasks**: Ingestion, preprocessing, chunking, embeddings, retrieval, generation, reranking, OCR
✅ **Non-functional**: Open-source stack, scalable, 0.4s latency, explainability, reproducibility
✅ **Deliverable**: Live demo with modern UI, citations, retrieval visualization, evaluation metrics

See `CHALLENGE_STATUS.md` for detailed completion status.

## Troubleshooting

### Groq API Not Working
- Check if API key is correct in `.env`
- Verify model name is valid (check Groq console)
- Falls back to Ollama if Groq fails

### Ollama Not Running
- Make sure Ollama is installed and running
- Check if Ollama is accessible: `curl http://localhost:11434/api/tags`
- Start Ollama: `ollama serve`

### PDF Ingestion Fails
- Check if PDF files are valid
- Ensure sufficient disk space
- Check file permissions

### No Results Retrieved
- Ensure PDFs have been ingested
- Check if index exists (faiss_index file)
- Try re-ingesting PDFs

### Slow Generation
- Use Groq API (recommended)
- Use a smaller/faster Ollama model
- Reduce chunk size
- Reduce TOP_K value
If you see model-related errors:
- Ensure llama3.2:1b is pulled: `ollama pull llama3.2:1b`
- Check the model name in `.env` file

### Memory Issues

If you encounter memory issues:
- Reduce batch size in `embeddings.py` (default: 32)
- Process fewer PDFs at a time
- Close other applications to free up RAM

### Tesseract Errors

Tesseract is optional. If you see Tesseract warnings:
- The app will still work for text-based PDFs
- Install Tesseract only if you need to process scanned PDFs
- Update the path in `.env` if installed in a non-default location

## Python 3.13 Compatibility

This project uses packages compatible with Python 3.13:
- sentence-transformers 3.3.1
- faiss-cpu 1.9.0.post1
- pypdfium2 4.30.0 (has pre-built wheels for Windows, avoiding compilation issues)
- numpy 2.1.3

All dependencies are specified in `requirements.txt` with tested versions.

## Performance Notes

- **Embedding Generation**: Depends on the number and size of PDFs. Typically 0.1-0.5 seconds per chunk.
- **Retrieval**: Very fast (<0.1 seconds for most queries)
- **Generation**: Depends on Ollama model and query complexity. Typically 1-5 seconds for llama3.2:1b.

## Requirements Status

### Completed Requirements

✅ Create all required files (app.py, ingestion.py, chunking.py, embeddings.py, vector_store.py, rag.py, evaluation.py, requirements.txt, README.md, .env.example)
✅ Streamlit interface for multiple PDF uploads
✅ Extract PDF text page by page with filename and page numbers
✅ Deterministic overlapping chunks
✅ FAISS CPU index with persistence
✅ Retrieve top 5 chunks
✅ Ollama local API with llama3.2:1b
✅ Show answer, citations, passages, similarity scores, latencies
✅ Error handling for missing dependencies
✅ Memory-efficient batched embeddings
✅ Python 3.13 compatible
✅ Evaluation script included
✅ Component testing script included
✅ Windows installation instructions
✅ All components tested and verified working

### Incomplete Requirements

❌ **At least 10 PDFs of at least 200 pages each**: This requires you to provide or download PDF documents. The application is ready to process any number of PDFs, but you need to supply the actual PDF files.

## Quick Start (PowerShell Commands)

Copy and paste these commands in PowerShell to get started:

```powershell
# Navigate to project directory
cd "c:\Users\taman\OneDrive\Desktop\rag-hackathon"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run component tests (optional, to verify everything works)
python test_components.py

# Start Ollama (in a separate PowerShell window)
ollama serve

# Run the Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## License

This project is created for the AI/ML hackathon.
