import os
import shutil
import uuid
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clip_service import clip_service

app = FastAPI(
    title="Semantic Video Search API",
    description="OpenCLIP-Powered Local Semantic Video Search API",
    version="0.3.0"
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

# Base uploads directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount uploads directory statically
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}


class SearchRequest(BaseModel):
    query: str
    video_id: str | None = None
    top_k: int = 6


def extract_frames_every_second(video_path: str, frames_dir: str):
    """
    Extracts 1 frame every second from the video using OpenCV.
    Saves frames into frames_dir as frame_0001.jpg, frame_0002.jpg, etc.
    Returns (total_extracted_frames, fps, timestamps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file with OpenCV")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or not fps or fps != fps:
        fps = 30.0

    frame_interval = max(1, int(round(fps)))
    
    frame_count = 0
    saved_frame_index = 1
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp_sec = round(frame_count / fps, 2)
            frame_filename = f"frame_{saved_frame_index:04d}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            
            cv2.imwrite(frame_path, frame)
            timestamps.append(timestamp_sec)
            saved_frame_index += 1

        frame_count += 1

    cap.release()

    return len(timestamps), round(fps, 2), timestamps


@app.get("/")
def read_root():
    return {
        "message": "OpenCLIP Semantic Video Search API is online",
        "version": "0.3.0",
        "status": "ready",
        "health_endpoint": "/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "semantic-video-search-backend",
        "clip_model": "OpenCLIP ViT-B-32",
        "version": "0.3.0"
    }


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    1. Receive local video file (.mp4, .mov, .avi)
    2. Save in uploads/{video_id}/
    3. Extract 1 frame per second into uploads/{video_id}/frames/
    4. Generate & store OpenCLIP embeddings into uploads/{video_id}/embeddings.npy
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Unique video folder
    video_id = str(uuid.uuid4())[:8]
    video_dir = os.path.join(UPLOAD_DIR, video_id)
    frames_dir = os.path.join(video_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    safe_filename = os.path.basename(file.filename)
    video_path = os.path.join(video_dir, safe_filename)

    try:
        # Save video file locally
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(video_path)

        # Step 1: Extract 1 frame per second using OpenCV
        total_frames, fps, timestamps = extract_frames_every_second(video_path, frames_dir)

        # Step 2: Generate & save OpenCLIP vector embeddings locally
        total_embeddings = clip_service.extract_and_save_embeddings(video_dir, frames_dir, timestamps)

        return {
            "status": "success",
            "message": "Video uploaded, frames extracted, and OpenCLIP embeddings indexed successfully",
            "video_id": video_id,
            "filename": safe_filename,
            "saved_path": f"uploads/{video_id}/{safe_filename}",
            "frames_dir": f"uploads/{video_id}/frames",
            "size_bytes": file_size,
            "fps": fps,
            "total_extracted_frames": total_frames,
            "indexed_embeddings_count": total_embeddings,
            "timestamps": timestamps
        }

    except Exception as e:
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")


@app.post("/search")
async def search_video(req: SearchRequest):
    """
    Search extracted video frames using natural language text query via OpenCLIP embeddings.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    try:
        results = clip_service.search(
            upload_base_dir=UPLOAD_DIR,
            query_text=req.query,
            video_id=req.video_id,
            top_k=req.top_k
        )

        return {
            "status": "success",
            "query": req.query,
            "video_id": req.video_id,
            "total_results": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search execution failed: {str(e)}")


@app.get("/search")
async def search_video_get(
    query: str = Query(..., description="Text query to search inside video"),
    video_id: str | None = Query(None, description="Optional specific video_id"),
    top_k: int = Query(6, description="Number of top matching frames to return")
):
    """
    GET version of semantic video search endpoint.
    """
    req = SearchRequest(query=query, video_id=video_id, top_k=top_k)
    return await search_video(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
