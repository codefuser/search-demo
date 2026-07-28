import os
import shutil
import uuid
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Semantic Video Search API",
    description="Prototype FastAPI backend for local semantic video search and frame extraction",
    version="0.2.0"
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
    if fps <= 0 or not fps or fps != fps:  # Check invalid or NaN
        fps = 30.0  # Default fallback FPS

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
    """Root endpoint delivering basic API information."""
    return {
        "message": "Semantic Video Search API is online",
        "version": "0.2.0",
        "status": "ready",
        "health_endpoint": "/health"
    }


@app.get("/health")
def health_check():
    """Health check endpoint to verify backend server status."""
    return {
        "status": "ok",
        "service": "semantic-video-search-backend",
        "version": "0.2.0"
    }


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Receive uploaded video, validate format (.mp4, .mov, .avi),
    save inside uploads/{video_id}/ and extract 1 frame every second into uploads/{video_id}/frames/
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{file_ext}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Generate unique video ID folder
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

        # Extract 1 frame per second using OpenCV
        total_frames, fps, timestamps = extract_frames_every_second(video_path, frames_dir)

        return {
            "status": "success",
            "message": "Video uploaded and frames extracted successfully",
            "video_id": video_id,
            "filename": safe_filename,
            "saved_path": f"uploads/{video_id}/{safe_filename}",
            "frames_dir": f"uploads/{video_id}/frames",
            "size_bytes": file_size,
            "fps": fps,
            "total_extracted_frames": total_frames,
            "timestamps": timestamps
        }

    except Exception as e:
        # Clean up directory if extraction failed
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
