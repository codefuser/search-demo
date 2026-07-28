# Semantic Video Search Prototype

A lightweight local prototype application for AI-Powered Semantic Video Search built with **React + Vite + TypeScript** for the frontend and **Python FastAPI** for the backend.

---

## 📁 Project Structure

```
search_demo_project/
├── frontend/                 # React + Vite + TypeScript Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header.tsx           # App navbar & backend health status
│   │   │   ├── VideoUploader.tsx    # Upload video button & drag-and-drop
│   │   │   ├── VideoPreview.tsx     # HTML5 video preview player
│   │   │   ├── SearchBar.tsx        # Disabled search textbox (prototype)
│   │   │   └── ResultsSection.tsx   # Empty search results container
│   │   ├── App.tsx                  # Main dashboard layout
│   │   ├── main.tsx                 # React DOM entry point
│   │   └── index.css                # Dark theme design system & styling
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── backend/                  # Python FastAPI Backend
│   ├── main.py               # FastAPI server with CORS & health endpoint
│   ├── requirements.txt      # Python backend dependencies
│   └── .gitignore
│
├── .gitignore
└── README.md
```

---

## 🚀 Running the Project

Frontend and backend run **independently** in separate terminal sessions.

### 1. Run Backend (FastAPI)

Open a terminal and navigate to `backend/`:

```bash
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
python main.py
```

The FastAPI server runs on `http://127.0.0.1:8000`.
- Health Endpoint: `http://127.0.0.1:8000/health`
- Interactive API Docs: `http://127.0.0.1:8000/docs`

---

### 2. Run Frontend (React + Vite + TypeScript)

Open a second terminal and navigate to `frontend/`:

```bash
cd frontend

# Install node dependencies
npm install

# Start Vite development server
npm run dev
```

The Vite dev server runs on `http://localhost:5173`.

---

## 📋 Features Included in Prototype

- **Frontend**:
  - React 18, Vite, TypeScript, Lucide Icons
  - Sleek dark theme with CSS custom properties & glassmorphic styling
  - Upload Video button with drag & drop file picker
  - Interactive Video preview player area with file info bar
  - Disabled Search Bar with prototype notice banner
  - Clean Empty State for Search Results
  - Real-time Backend Health Connection indicator in Header

- **Backend**:
  - Python FastAPI web application
  - Enabled CORS middleware allowing frontend requests
  - Health check endpoint (`GET /health`)
