from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
