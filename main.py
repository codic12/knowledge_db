import os
import glob
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from ocr_engine import process_file
from vectorless_engine import vectorless_index
from llm_client import generate_answer

app = FastAPI(title="Document Intelligence API")

# Mount static files for UI
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class IngestRequest(BaseModel):
    s3_path: str
    cascade: bool = False

class AskRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/files/{path:path}")
async def get_file(path: str):
    """
    Endpoint to serve documents for preview.
    Note: In a real S3 scenario, this would generate a pre-signed URL.
    """
    if os.path.exists(path) and os.path.isfile(path):
        return FileResponse(path)
    raise HTTPException(status_code=404, detail="File not found")

def perform_ingestion(folder_path: str, cascade: bool):
    """
    Background task to process files and build index.
    """
    search_pattern = os.path.join(folder_path, "**/*") if cascade else os.path.join(folder_path, "*")
    
    files_to_process = []
    for filepath in glob.glob(search_pattern, recursive=cascade):
        if os.path.isfile(filepath):
            files_to_process.append(filepath)
            
    print(f"Found {len(files_to_process)} files to process in {folder_path} (cascade={cascade})")
    
    for file_path in files_to_process:
        vectorless_index.add_document(file_path)

@app.post("/api/ingest")
async def ingest_documents(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    1. Document Ingestion API
    """
    if not os.path.exists(req.s3_path) or not os.path.isdir(req.s3_path):
        raise HTTPException(status_code=400, detail=f"Path {req.s3_path} does not exist or is not a directory.")
        
    background_tasks.add_task(perform_ingestion, req.s3_path, req.cascade)
    return {"status": "success", "message": f"Ingestion started for {req.s3_path}."}

@app.post("/api/ask")
async def ask_question(req: AskRequest):
    """
    2. Question Answering API
    """
    question = req.question
    
    # Retrieve top relevant context (vectorless only)
    top_contexts = vectorless_index.search(question, top_n=5)
    
    # Generate answer (using NVIDIA NIM via llm_client)
    answer = generate_answer(question, top_contexts)
    
    # Format citations with full paths for the UI to use
    citations = []
    for ctx in top_contexts:
        citations.append({
            "document": os.path.basename(ctx['filepath']),
            "filepath": ctx['filepath'],
            "page": ctx['page']
        })
        
    return {
        "answer": answer,
        "citations": citations
    }
