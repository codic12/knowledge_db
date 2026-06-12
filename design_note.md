# PayGlocal Enterprise Document Intelligence - Design Note

## Architecture & Approach
This prototype is built using a modern, scalable, yet cost-efficient architecture to process and retrieve information from a highly varied set of documents.

### Components
1. **API Layer (FastAPI)**: Provides highly performant, asynchronous endpoints for ingestion and querying.
2. **OCR & Image Processing Engine (`ocr_engine.py`)**: 
   - Utilizes `pdf2image` to standardize PDF inputs into images.
   - Leverages `OpenCV` and `Pillow` for preprocessing: grayscale conversion and automatic deskewing/rotation based on coordinate analysis and Tesseract OSD (Orientation and Script Detection). This effectively handles skewed or rotated documents.
   - Utilizes `Tesseract OCR` to extract text from images, scanned pages, and basic handwritten content.
3. **Retrieval Engine (`rag_engine.py`)**: 
   - Implements **Hybrid Search** using a combination of **Vector Search (FAISS)** and **Lexical Search (BM25)**.
   - Utilizes `EnsembleRetriever` from LangChain to merge results from both strategies.
   - This approach solves the "Keyword Gap" where semantic vector embeddings often miss specific alphanumeric identifiers like section numbers (e.g., "6a").
   - Utilizes `all-MiniLM-L6-v2` HuggingFace embeddings for the semantic component.
   - Fits well within the 16GB RAM limit for a large number of documents.
4. **Generation Engine (`llm_client.py`)**: 
   - Uses an economical API via Nvidia NIM (e.g., Llama3 70B or Qwen equivalents).
   - Instructed with a strict system prompt to heavily penalize hallucination and force citations.

## Vector-based RAG with LangChain
While the previous iteration explored BM25 (vectorless), this version adopts an industry-standard **Vector-based RAG** approach.

**Why Vector RAG for this use-case?**
1. **Semantic Understanding**: Vector embeddings capture the meaning behind words, allowing the system to find relevant sections even if the user doesn't use the exact keywords found in the document.
2. **Scalability & Framework Support**: By using established frameworks like `LangChain` and `FAISS`, the system can easily scale to handle thousands of documents with efficient nearest-neighbor searches.
3. **Local Embeddings**: Using `sentence-transformers` locally ensures zero per-query embedding costs and keeps data private within the compute environment.

## Citation Strategy
Citations remain deterministic and are preserved through LangChain metadata:
1. **Granularity**: The `ocr_engine` segments text extraction strictly by document and page number. 
2. **Metadata Injection**: Each chunk is stored in FAISS with metadata containing `filepath` and `page`.
3. **Context Passing**: When a user queries, the top semantic matches are retrieved. Each chunk of text passed to the LLM is prefixed with `--- Document: {filename}, Page: {page} ---`.
3. **LLM Formatting**: The LLM is strictly prompted to append `[DocumentName, Page X]` to any claim it makes based on the context.
4. **UI Highlighting**: The UI surfaces these exact documents and pages in a dedicated "Retrieved Context" section below the answer, ensuring total transparency.

## Handling Visual & Complex Information
- **Scans & Low Quality**: OpenCV adaptive thresholding and grayscale conversion cleans up noise before passing it to Tesseract.
- **Skewed/Rotated**: Tesseract's `image_to_osd` detects rotation angles, and OpenCV `minAreaRect` helps deskew slight angles, ensuring OCR doesn't misread vertical/diagonal text.
- **Handwriting**: Tesseract (especially with fine-tuned models) can extract legible handwriting. For future scale, adding a lightweight Vision-Language Model (VLM) step just for handwritten segments could enhance accuracy.

## Refusal to Answer
The system uses an explicit "Refusal" strategy. If the top BM25 results do not contain the answer, the LLM prompt explicitly commands: *"If the answer is not in the context, say 'I cannot answer this based on the provided documents.'"* It will not fallback to its pre-trained knowledge.

## Brief Evaluation Framework
- **Accuracy**: Measured by injecting known facts into mock documents and testing if the system retrieves them exactly and attributes the correct page.
- **Latency**: 
  - *Ingestion*: Bounded by OCR time per page (~1-2 seconds per page).
  - *Retrieval*: BM25 retrieval is sub-100ms.
  - *Generation*: Llama3/Qwen inference via NIM typically runs at 50-100 tokens/second, resulting in end-to-end QA latency of ~2-4 seconds.
- **Cost**: 
  - Embedding Cost: $0 (Vectorless).
  - LLM Cost: Extremely low utilizing open-weights models via Nvidia NIM compared to GPT-4o.
