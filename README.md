# Enterprise Document Intelligence Prototype

## Setup Instructions

1. **Setup a Virtual Environment** (Highly Recommended on macOS/Linux):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install System Dependencies**:
   - **Tesseract OCR**: `brew install tesseract`
   - **Poppler**: `brew install poppler`

3. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variable**:
   ```bash
   export NVIDIA_API_KEY="your_api_key_here"
   ```

5. **Run the Application**:
   ```bash
   uvicorn main:app --reload
   ```

6. **Test the System**:
   - Open your browser to `http://127.0.0.1:8000/`.
   - You will see the ChatGPT-like UI.
   - Click "Ingest Documents" to ingest the provided `mock_s3/sample_contract.txt`.
   - Ask questions like: "What is the overage cost?" or "Who is the contact for Acme Corp?"

## Features Implemented:
- **Python + FastAPI**: Backend framework.
- **Tesseract OCR**: Image and PDF processing with automatic deskewing.
- **Nvidia NIM**: LLM inference using `meta/llama3-70b-instruct` for cost-efficient generation.
- **Vector-based RAG**: Utilizes **LangChain** and **FAISS** with `all-MiniLM-L6-v2` local embeddings for semantic retrieval.
- **Mock S3**: Ingestion cascades through local directories simulating S3 paths.
- **Beautiful HTML UI**: Clean, ChatGPT-inspired interface.
- **Strict Citations**: System refuses to hallucinate and cites exact Document + Page.
