import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Semantic Video Search API",
    description="Prototype FastAPI backend for local semantic video search application",
    version="0.1.0"
)

# Enable CORS for frontend client interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure uploads directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Optionally mount uploads directory to serve uploaded videos statically
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}


@app.get("/")
def read_root():
    """Root endpoint delivering basic API information."""
    return {
        "message": "Semantic Video Search API is online",
        "version": "0.1.0",
        "status": "ready",
        "health_endpoint": "/health"
    }


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend server status."""
    return {
        "status": "ok",
        "service": "semantic-video-search-backend",
        "version": "0.1.0"
    }


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Receive uploaded video from frontend, validate format (.mp4, .mov, .avi),
    and save inside backend/uploads/ directory.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Sanitize filename
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(file_path)

        return {
            "status": "success",
            "message": "Video uploaded successfully",
            "filename": safe_filename,
            "saved_path": f"uploads/{safe_filename}",
            "size_bytes": file_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
