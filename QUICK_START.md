# Quick Start Guide - RAG PDF Chatbot

## 1. Installation (2 minutes)

```bash
# Navigate to project directory
cd rag-hackathon

# Activate virtual environment
venv\Scripts\Activate

# Dependencies are already installed
# If needed: pip install -r requirements.txt
```

## 2. Run the Application (30 seconds)

```bash
streamlit run app.py
```

The app will open at: **http://localhost:8501**

## 3. Demo Workflow (5 minutes)

### Step 1: Ingest PDFs
1. Click **"Ingest PDFs"** tab
2. Click **"📁 Choose PDF files"**
3. Select PDFs from `test_pdfs/` folder (12 large PDFs available)
4. Click **"🚀 Ingest PDFs"**
5. Wait for processing (~2-3 minutes for 12 PDFs)

### Step 2: Ask Questions
1. Click **"Chat"** tab
2. Try these questions:
   - "What is machine learning?"
   - "Explain neural networks."
   - "What is deep learning?"
   - "How does computer vision work?"
3. View answers with citations and metrics

## 4. Key Features to Demonstrate

### ✅ Sub-Second Latency
- Total response time: ~0.4 seconds
- Well within 2-5 second requirement

### ✅ Source Citations
- Each answer includes PDF filename + page number
- Click "View Retrieved Passages" to see source chunks

### ✅ Performance Metrics
- Retrieval time (~0.01s)
- Generation time (~0.39s)
- Total time (~0.40s)

### ✅ Modern UI
- ChatGPT-style chat interface
- Dark theme with gradient headers
- Real-time metrics display

## 5. Technical Highlights

### Architecture
- **Modular Design**: Separate files for each component
- **Dual LLM**: Groq API (fast) + Ollama (fallback)
- **Open-Source**: sentence-transformers + FAISS

### Preprocessing
- Header/footer removal
- Language detection
- Text cleaning

### Evaluation
- Recall@K (R@5, R@10)
- Mean Reciprocal Rank (MRR)
- Citation accuracy

## 6. Run Evaluation (Optional)

```bash
python evaluation.py
```

This tests the system with sample queries and shows metrics.

## 7. Configuration

Edit `.env` file to customize:
- Groq API key
- Chunk size (default: 512)
- Chunk overlap (default: 128)
- Top-K retrieval (default: 5)
- Reranking toggle (default: false)

## 8. Troubleshooting

**Issue**: "Ollama not running"
- **Solution**: System uses Groq API by default, Ollama is only fallback

**Issue**: "Groq API error"
- **Solution**: Check API key in `.env` file

**Issue**: "No index found"
- **Solution**: Ingest PDFs first in "Ingest PDFs" tab

---

**For full documentation**: See `README.md` and `HACKATHON_SUBMISSION.md`
