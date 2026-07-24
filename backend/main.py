"""V-RAG Backend — FastAPI application."""

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.services.transcript import load_transcript
from backend.services.pdf import extract_text_from_pdf
from backend.services.github_service import load_repo
from backend.services.rag import process_text, process_video, process_repo, chat, get_session, delete_session

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="V-RAG API",
    description="YouTube Video, PDF & GitHub Repo RAG Assistant — backend API",
    version="2.0.0",
)

# Allow the Streamlit frontend (and local dev) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Max PDF upload size (10 MB)
MAX_PDF_SIZE_MB = 10


# ─── Request / Response models ────────────────────────────────────────────────


class ProcessRequest(BaseModel):
    video_url: str
    hf_api_token: str
    supadata_api_key: str | None = None


class ProcessResponse(BaseModel):
    session_id: str
    summary: str


class ChatRequest(BaseModel):
    session_id: str
    question: str
    hf_api_token: str


class ChatResponse(BaseModel):
    answer: str


class HealthResponse(BaseModel):
    status: str


class RepoProcessRequest(BaseModel):
    repo_url: str
    hf_api_token: str
    github_token: str | None = None


class RepoProcessResponse(BaseModel):
    session_id: str
    summary: str
    file_count: int


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for Render."""
    return HealthResponse(status="ok")


# ─── YouTube video processing ────────────────────────────────────────────────


@app.post("/api/process", response_model=ProcessResponse)
def process_video_endpoint(req: ProcessRequest):
    """
    Process a YouTube video:
    1. Fetch transcript (Supadata → youtube-transcript-api fallback)
    2. Build FAISS vector store
    3. Generate summary
    Returns a session_id for subsequent chat calls.
    """
    try:
        transcript_text = load_transcript(
            req.video_url,
            supadata_key=req.supadata_api_key,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Transcript error: {str(e)}")

    try:
        session_id, summary = process_video(transcript_text, req.hf_api_token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return ProcessResponse(session_id=session_id, summary=summary)


# ─── PDF processing ──────────────────────────────────────────────────────────


@app.post("/api/process-pdf", response_model=ProcessResponse)
async def process_pdf_endpoint(
    file: UploadFile = File(...),
    hf_api_token: str = Form(...),
):
    """
    Process an uploaded PDF:
    1. Extract text from PDF
    2. Build FAISS vector store
    3. Generate summary
    Returns a session_id for subsequent chat calls.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Read and validate size
    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"PDF too large ({size_mb:.1f} MB). Maximum is {MAX_PDF_SIZE_MB} MB.",
        )

    # Extract text
    try:
        pdf_text = extract_text_from_pdf(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF extraction error: {str(e)}")

    # Build RAG pipeline + summarize
    try:
        session_id, summary = process_text(pdf_text, hf_api_token, source_type="pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return ProcessResponse(session_id=session_id, summary=summary)


# ─── GitHub repo processing ───────────────────────────────────────────────


@app.post("/api/process-repo", response_model=RepoProcessResponse)
def process_repo_endpoint(req: RepoProcessRequest):
    """
    Process a GitHub repository:
    1. Download repo as zip via GitHub API
    2. Parse and filter code files
    3. Build FAISS vector store with file-path metadata
    4. Generate project overview summary
    Returns a session_id for subsequent chat calls.
    """
    try:
        files = load_repo(req.repo_url, github_token=req.github_token)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Repo error: {str(e)}")

    try:
        session_id, summary, file_count = process_repo(files, req.hf_api_token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    return RepoProcessResponse(
        session_id=session_id, summary=summary, file_count=file_count
    )


# ─── Chat (works for video, PDF, and repo sessions) ───────────────────────


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Ask a question against a processed session (video, PDF, or repo)."""
    session = get_session(req.session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Please process the source again.",
        )

    try:
        answer = chat(req.session_id, req.question, req.hf_api_token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

    return ChatResponse(answer=answer)


@app.delete("/api/session/{session_id}")
def delete_session_endpoint(session_id: str):
    """Clear a session and its resources."""
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"detail": "Session deleted."}
