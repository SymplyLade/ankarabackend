"""
backend/main.py - Render-ready AnkaraPrint RAG API
"""
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from rag_system import create_default_rag  # Uses AnkaraPrintRAGSystem

# Global RAG system reference (lazy loaded)
rag_system = None

app = FastAPI()


@app.get("/status")
def status():
    return {
        "ready": rag_system is not None
    }


rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system
    print("Initializing RAG system...")

    try:
        rag_system = create_default_rag()
        print("RAG system loaded successfully.")
    except Exception as e:
        print("RAG failed to load:", e)
        rag_system = None  # allow app to run anyway

    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)


# FastAPI app
app = FastAPI(
    title="AnkaraPrint RAG API",
    description="Render-ready Simplified API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------
# Models
# -------------------
class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]

class SystemStatus(BaseModel):
    status: str
    pdf_loaded: bool
    vector_db_ready: bool
    llm_connected: bool
    total_chunks: int

# -------------------
# Endpoints
# -------------------
@app.get("/")
async def root():
    return {
        "message": "AnkaraPrint Simplified RAG API",
        "version": "2.0.0",
        "endpoints": {
            "chat": "POST /api/chat",
            "status": "GET /api/status",
            "upload": "POST /api/upload-pdf",
            "profile": "GET /api/profile"
        }
    }

@app.get("/api/status")
async def get_status():
    global rag_system
    if rag_system is None:
        return SystemStatus(
            status="offline",
            pdf_loaded=False,
            vector_db_ready=False,
            llm_connected=False,
            total_chunks=0
        )
    status = rag_system.get_system_status()
    return SystemStatus(
        status="online",
        pdf_loaded=status["pdf_loaded"],
        vector_db_ready=status["vector_db_ready"],
        llm_connected=status["llm_connected"],
        total_chunks=status["total_chunks"]
    )

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    global rag_system
    if rag_system is None:
        rag_system = create_default_rag()  # lazy load to save memory
    try:
        response, sources = rag_system.get_response(message.message)
        return ChatResponse(response=response, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global rag_system
    if rag_system is None:
        rag_system = create_default_rag()
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    try:
        temp_path = f"temp_{file.filename}"
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        chunks_created = rag_system.process_pdf(temp_path)
        os.remove(temp_path)

        return {"success": True, "message": f"PDF processed: {file.filename}", "chunks_created": chunks_created}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/api/profile")
async def get_profile():
    return {
        "name": "AnkaraPrint Project",
        "description": "Learning and exploring Ankara Printing techniques",
        "contact": {
            "email": "contact@ankaraprint.com",
            "linkedin": "https://www.linkedin.com/company/ankaraprint",
            "portfolio": "https://ankaraprint.com"
        }
    }

@app.get("/api/test")
async def test_endpoint():
    return {"status": "ok", "rag_system_ready": rag_system is not None}

# -------------------
# Run on Render
# -------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))  # Use Render's port
    print(f"\nStarting FastAPI server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)  # remove --reload for production

