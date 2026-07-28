# 🎬 Semantic Video Search Project - Comprehensive Technical Report

---

## 📌 1. Project Overview

The **Semantic Video Search Application** is a local, AI-powered web application prototype designed to perform natural language semantic search inside video content. 

Users can upload any local video file (`.mp4`, `.mov`, `.avi`), and the application automatically extracts 1-second frames, generates 512-dimensional vision-language vector embeddings locally using a pretrained **CLIP model**, and allows users to search for moments, objects, or actions (e.g., *"red shirt"*, *"white shoes"*, *"car"*, *"human"*, *"phone"*) without requiring any cloud APIs, external databases, or object detection models (YOLO).

---

## 📂 2. Repository & Project Structure

- **GitHub Repository**: [`https://github.com/codefuser/search-demo.git`](https://github.com/codefuser/search-demo.git)
- **Local Directory**: `D:\Final_year_project\search_demo_project`

```
search_demo_project/
├── frontend/                     # React + Vite + TypeScript Client
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx        # Top navbar & live API connection status
│   │   │   ├── VideoUploader.tsx # Drag & drop file picker + Indexing progress bar
│   │   │   ├── VideoPreview.tsx  # Interactive HTML5 video player with timestamp seek
│   │   │   ├── SearchBar.tsx     # Text search bar, Enter key listener & search history
│   │   │   └── ResultsSection.tsx# Ranked Top 20 result cards with score & timestamps
│   │   ├── App.tsx               # Main state manager & backend API integrator
│   │   ├── main.tsx              # React DOM entry
│   │   └── index.css             # Dark theme design system & styling
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── backend/                      # Python FastAPI Server
│   ├── main.py                   # FastAPI app routes (/upload, /search, /health)
│   ├── clip_service.py           # CLIP model wrapper, RAM caching & vector search
│   ├── requirements.txt          # Python backend dependencies
│   ├── uploads/                  # Local storage for videos, frames & embeddings
│   └── .gitignore
│
├── .gitignore
├── README.md                     # Setup and execution guide
└── PROJECT_REPORT.md             # Complete implementation report
```

---

## 🚀 3. Features Implemented (Phase by Phase)

### Phase 1: Core Architecture & Setup
- Built a modular decoupled architecture: **React + Vite + TypeScript** frontend running on port `5173`, and **Python FastAPI** backend running on port `8000`.
- Configured CORS middleware to allow seamless local cross-origin communication between client and server.
- Initialized local Git repository, created `main` branch, and connected to GitHub repository `codefuser/search-demo.git`.

### Phase 2: Local Video Upload & Metadata Player
- Implemented file upload dropzone supporting `.mp4`, `.mov`, and `.avi` video formats.
- Created interactive HTML5 video preview player extracting file metadata: **Filename**, formatted **File Size** (KB/MB), and calculated **Duration** (`MM:SS` / `HH:MM:SS`).
- Configured backend storage creating unique video folders inside `backend/uploads/{video_id}/`.

### Phase 3: Frame Extraction Engine (OpenCV)
- Implemented 1-frame-per-second video sampling using OpenCV (`cv2.VideoCapture`).
- Saves extracted frames into `backend/uploads/{video_id}/frames/` named sequentially (`frame_0001.jpg`, `frame_0002.jpg`, etc.).
- Includes automatic frame extraction reuse: skips re-extracting if frame images already exist.

### Phase 4: Pretrained Vision-Language Semantic Search Engine (CLIP)
- Integrated pretrained **HuggingFace CLIP (`openai/clip-vit-base-patch32`)** model.
- **100% Offline & Local**: No external cloud APIs, no database, no dataset training, and no object detection / YOLO.
- **Vector Embedding Generation**: Preprocesses each frame image and extracts normalized 512-dimensional vector representations stored in `embeddings.npy` alongside `metadata.json`.
- **Cosine Similarity Dot Product**: Converts user text query into text embedding vector and calculates dot product similarity score ($\mathbf{v}_{\text{image}} \cdot \mathbf{v}_{\text{text}}$).
- **Ranking**: Sorts search results descending by similarity score (highest match first).

### Phase 5: RAM Embedding Caching & Performance Optimization
- **In-Memory RAM Cache (`_embeddings_cache`)**: Loads embeddings into RAM **only once** per video ID.
- **Zero-Disk Matrix Search**: Searches execute pure in-memory matrix multiplication (`np.dot`) with 0 disk re-reading.
- **Deterministic Hash Signature (`MD5`)**: Re-uploading the same video file resolves to the same `video_id`, skipping extraction and embedding calculations instantly.

### Phase 6: Result Cards UI & Click-to-Play Video Seeking
- **Top 20 Matches**: Displays up to Top 20 matching frame cards sorted by score.
- **Card Metadata**: Displays frame thumbnail image, exact timestamp (`00:12 (12s)`), and similarity score (e.g., `Score: 0.3125`).
- **Interactive Video Seeking**: Clicking any search result card instantly seeks and plays the video preview player at that exact timestamp second.
- **Active Card Highlight**: Clicked card receives a highlighted primary indigo border (`2px solid #6366f1`), glowing shadow, and `Playing` status badge.
- **Search History (`localStorage`)**: Persists recent search queries (e.g., `red shirt`, `white shoes`, `car`, `human`) as clickable chips for 1-click re-searching, with a "Clear History" button.
- **Indexing Progress Bar**: Visual progress bar (`0%` $\rightarrow$ `100%`) displaying frame extraction and vector indexing progress.
- **Keyboard Support**: Pressing **Enter** in the search input triggers search natively.

### Phase 7: Bug Fixes & Stability
- Resolved Windows `Python was not found` PATH issue by installing Python 3.11 with PATH configuration.
- Resolved `RuntimeError: operator torchvision::nms does not exist` C++ operator issue by migrating from `open-clip-torch`/`torchvision` to HuggingFace `transformers` CLIP model pipeline.

---

## 📊 4. API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check & returns server status, model info, and RAM cached video count |
| `POST` | `/upload` | Receives video file, extracts 1-second frames, generates & caches CLIP embeddings |
| `POST` | `/search` | Accepts `{ "query": "red shirt", "video_id": "...", "top_k": 20 }` and returns ranked results |
| `GET` | `/search` | Query param version of semantic search endpoint |

---

## 🏃 5. How to Run the Application

### 1. Start Backend Server (Terminal 1)
```bash
cd backend
venv\Scripts\activate
python main.py
```
*Backend runs on `http://127.0.0.1:8000`*

### 2. Start Frontend Application (Terminal 2)
```bash
cd frontend
npm run dev
```
*Frontend runs on `http://localhost:5173`*

---

## 🏁 6. Conclusion

The **Semantic Video Search Prototype** is complete, fully optimized with RAM embedding caching, features clean minimal UI design with video seeking, and runs 100% locally offline. All code updates are committed and pushed to GitHub.
