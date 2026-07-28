import os
import shutil
import glob
import hashlib
import json
import time
import asyncio
from contextlib import asynccontextmanager
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from clip_service import clip_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan context manager:
    Pre-loads the CLIP model into memory ONCE during server startup.
    This guarantees that model loading NEVER blocks or hangs user search requests!
    """
    print("=================================================================")
    print("[SERVER STARTUP] Pre-loading CLIP model into memory...")
    t0 = time.perf_counter()
    clip_service.load_model()
    elapsed = time.perf_counter() - t0
    print(f"[SERVER STARTUP] CLIP model successfully loaded in {elapsed:.3f}s!")
    print("=================================================================\n")
    yield
    print("[SERVER SHUTDOWN] Shutting down backend server...")


app = FastAPI(
    title="Semantic Video Search API",
    description="High-performance cached CLIP local semantic video search API",
    version="0.6.0",
    lifespan=lifespan
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
    top_k: int = 20


def extract_frames_every_second(video_path: str, frames_dir: str):
    """
    Extracts 1 frame every second from the video using OpenCV.
    OPTIMIZATION: Skips extraction if frames already exist in frames_dir.
    """
    t0_ext = time.perf_counter()
    existing_frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    
    if len(existing_frames) > 0:
        print(f"[STAGE 2 - Frame Extraction] Found {len(existing_frames)} existing frames. Skipping extraction (0.00s).")
        
        metadata_path = os.path.join(os.path.dirname(frames_dir), "metadata.json")
        timestamps = []
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
                timestamps = [m.get("timestamp", float(i)) for i, m in enumerate(meta_data)]
        
        if not timestamps:
            timestamps = [float(i) for i in range(len(existing_frames))]

        return len(existing_frames), 30.0, timestamps, True

    print(f"[STAGE 2 - Frame Extraction] Starting OpenCV extraction from '{video_path}'...")
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
    ext_elapsed = time.perf_counter() - t0_ext
    print(f"[STAGE 2 - Frame Extraction] Extracted {len(timestamps)} frames @ {round(fps,2)} FPS in {ext_elapsed:.3f}s.")

    return len(timestamps), round(fps, 2), timestamps, False


@app.get("/")
def read_root():
    return {
        "message": "CLIP Semantic Video Search API (Pre-loaded Lifespan Edition)",
        "version": "0.6.0",
        "status": "ready",
        "health_endpoint": "/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "semantic-video-search-backend",
        "clip_model": "CLIP ViT-B-32",
        "model_preloaded": clip_service._is_loaded,
        "ram_cached_videos_count": len(clip_service._embeddings_cache),
        "version": "0.6.0"
    }


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    1. Validates file
    2. Hashes video signature for instant cache lookup
    3. Runs frame extraction & embedding generation in worker threadpool (non-blocking)
    """
    t_upload_start = time.perf_counter()
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    safe_filename = os.path.basename(file.filename)
    
    file_bytes = await file.read(1024 * 1024)
    file_signature = f"{safe_filename}_{len(file_bytes)}"
    video_id = hashlib.md5(file_signature.encode()).hexdigest()[:10]
    
    await file.seek(0)

    video_dir = os.path.join(UPLOAD_DIR, video_id)
    frames_dir = os.path.join(video_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    video_path = os.path.join(video_dir, safe_filename)

    try:
        if not os.path.exists(video_path):
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(video_path)

        # Run CPU work in worker threadpool using asyncio.to_thread
        total_frames, fps, timestamps, is_frames_cached = await asyncio.to_thread(
            extract_frames_every_second, video_path, frames_dir
        )

        total_embeddings, is_embeddings_cached = await asyncio.to_thread(
            clip_service.extract_and_save_embeddings, video_dir, frames_dir, timestamps
        )

        total_upload_time = time.perf_counter() - t_upload_start
        cache_tag = "Retrieved from cache (instant)" if (is_frames_cached and is_embeddings_cached) else "Newly indexed"

        print(f"[UPLOAD COMPLETE] Video '{safe_filename}' (ID: {video_id}) processed in {total_upload_time:.3f}s!\n")

        return {
            "status": "success",
            "message": f"Indexing complete! {cache_tag}",
            "video_id": video_id,
            "filename": safe_filename,
            "saved_path": f"uploads/{video_id}/{safe_filename}",
            "frames_dir": f"uploads/{video_id}/frames",
            "size_bytes": file_size,
            "fps": fps,
            "total_extracted_frames": total_frames,
            "indexed_embeddings_count": total_embeddings,
            "is_cached": (is_frames_cached and is_embeddings_cached),
            "processing_time_sec": round(total_upload_time, 3),
            "timestamps": timestamps
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")


@app.post("/search")
async def search_video(req: SearchRequest):
    """
    Search extracted video frames using natural language text query.
    Executes search in worker thread with timeout protection (10s timeout).
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty")

    try:
        # Wrap CPU/matrix computation in asyncio.to_thread with 10.0s timeout protection
        results = await asyncio.wait_for(
            asyncio.to_thread(
                clip_service.search,
                UPLOAD_DIR,
                req.query,
                req.video_id,
                req.top_k
            ),
            timeout=10.0
        )

        return {
            "status": "success",
            "query": req.query,
            "video_id": req.video_id,
            "total_results": len(results),
            "results": results
        }
    except asyncio.TimeoutError:
        print(f"[SEARCH TIMEOUT ERROR] Query '{req.query}' timed out after 10.0 seconds!")
        raise HTTPException(status_code=504, detail="Search request timed out after 10 seconds.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search execution failed: {str(e)}")


@app.get("/search")
async def search_video_get(
    query: str = Query(..., description="Text query to search inside video"),
    video_id: str | None = Query(None, description="Optional specific video_id"),
    top_k: int = Query(20, description="Number of top matching frames to return (default 20)")
):
    req = SearchRequest(query=query, video_id=video_id, top_k=top_k)
    return await search_video(req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
