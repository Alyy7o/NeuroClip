# NeuroClip - Complete Setup Guide

A semantic multimodal search engine for transcribed video datasets with AI-powered video processing capabilities.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Project Structure](#project-structure)
- [Running the Application](#running-the-application)
- [Environment Configuration](#environment-configuration)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

NeuroClip is a full-stack application that enables:

- *Semantic Video Search*: Search through video datasets using natural language queries
- *Video Summarization*: AI-powered extraction of key highlights based on custom queries
- *Video Blurring*: Intelligent blurring of faces, objects, or sensitive content
- *Video Compression*: Smart compression maintaining quality while reducing file size
- *Multimodal Search*: Combines both text transcriptions and frame descriptions for intelligent search

### Technology Stack

*Backend:*
- Node.js + Express.js (server)
- Python (data processing, AI/ML)
- Supabase (vector database)
- AssemblyAI (transcription)
- yt-dlp (video download)
- FFmpeg (video processing)

*Frontend:*
- React 18 + TypeScript
- Vite (build tool)
- Tailwind CSS + shadcn/ui
- Supabase (authentication & database)
- Framer Motion (animations)
- React Query (data fetching)

---

## 📦 Prerequisites

Before you begin, ensure you have the following installed:

### Required Software

- *Node.js*: v18 or higher
  - Download from https://nodejs.org/
  - Verify: node --version

- *npm*: v9 or higher (comes with Node.js)
  - Verify: npm --version

- *Python*: v3.8 or higher
  - Download from https://www.python.org/
  - Verify: python --version

- *Git*: For version control
  - Download from https://git-scm.com/

### Optional but Recommended

- *Visual Studio Code*: Code editor
- *Postman*: API testing tool
- *FFmpeg*: For video processing (install if not already present)

### External API Keys/Services

You'll need to set up accounts and obtain API keys for:

- *AssemblyAI*: For video transcription
  - Sign up at https://www.assemblyai.com/
  - Get your API key from the dashboard



- *Supabase*: For authentication and database
  - Sign up at https://supabase.com/
  - Create a new project and get credentials

---

## 🔧 Backend Setup

### Step 1: Navigate to Backend Directory

powershell
cd backend


### Step 2: Install Python Dependencies

The backend requires Python packages. Install them using:

powershell
pip install -r requirements.txt


*Note*: The requirements.txt includes GPU support (CUDA 11) for PyTorch. If you don't have an NVIDIA GPU or encounter issues:

powershell
# For CPU-only version, create a virtual environment first
python -m venv venv
.\venv\Scripts\Activate.ps1

# Then install dependencies
pip install -r requirements.txt


#### Key Python Dependencies

- *torch, torchvision*: Deep learning framework
- *transformers, sentence-transformers*: NLP and embedding models
- *nltk*: Natural language processing
- *scikit-learn*: Machine learning utilities
- *weaviate-client*: Vector database client
- *assemblyai*: Transcription API client

### Step 3: Install Node.js Dependencies

The backend also uses Node.js for the server:

powershell
npm install


#### Key Node.js Dependencies

- *express*: Web server framework
- *cors*: Cross-Origin Resource Sharing
- *fluent-ffmpeg*: Video processing
- *assemblyai*: Transcription SDK
- *yt-dlp-wrap*: YouTube video download
- *dotenv*: Environment variable management
- *uuid*: Unique identifier generation

### Step 4: Set Up Environment Variables

Create a .env file in the backend directory:

powershell
# Copy the example or create new
Copy-Item .env.example .env


Add the following environment variables:

env
# AssemblyAI Configuration
ASSEMBLYAI_API_KEY=your_assemblyai_api_key_here

# Weaviate Configuration
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your_weaviate_api_key_here

# Server Configuration
PORT=5000
NODE_ENV=development

# Video Processing
TEMP_DIR=./temp
OUTPUT_DIR=./output_data

# Optional: YouTube DL Configuration
YT_DLP_PATH=yt-dlp


### Step 5: Verify Installation

Test that all dependencies are correctly installed:

powershell
# Test Python
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Test Node.js
node --version
npm --version


---

## 🚀 Frontend Setup

### Step 1: Navigate to Frontend Directory

From the project root:

powershell
cd frontend


### Step 2: Install Dependencies

The frontend uses *bun* as a package manager (optional but recommended):

powershell
# Using npm
npm install

# OR using bun (faster)
bun install


#### Key Frontend Dependencies

- *react, react-dom*: UI framework
- *react-router-dom*: Client-side routing
- *@supabase/supabase-js*: Supabase client
- *@tanstack/react-query*: Server state management
- *framer-motion*: Animation library
- *react-player*: Video player component
- *recharts*: Data visualization
- *tailwindcss, shadcn/ui*: UI framework and components
- *zod*: TypeScript-first schema validation

### Step 3: Set Up Environment Variables

Create a .env.local file in the frontend directory:

powershell
# Copy the example or create new
Copy-Item .env.example .env.local


Add the following environment variables:

env
# Supabase Configuration
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key

# Backend API Configuration
VITE_API_URL=http://localhost:5000

# App Configuration
VITE_APP_NAME=NeuroClip
VITE_APP_URL=http://localhost:5173


### Step 4: Get Supabase Credentials

1. Go to https://supabase.com/ and sign in
2. Create a new project
3. Navigate to Settings → API
4. Copy:
   - Project URL → VITE_SUPABASE_URL
   - anon (public) key → VITE_SUPABASE_ANON_KEY

### Step 5: Build Configuration

The frontend is configured with Vite for optimal development experience:

powershell
# Development build (default)
npm run build

# Production build
npm run build:dev

# Preview the built app
npm run preview


---

## 📂 Project Structure


NeuroClip/
├── backend/                          # Backend services
│   ├── server.js                     # Express server (main entry)
│   ├── requirements.txt              # Python dependencies
│   ├── assemblyai_utils.py          # Transcription utilities
│   ├── test_transcribe.py           # Test script for transcription
│   ├── weaviate_data_populate.ipynb # Data population notebook
│   ├── input_files/                  # Video input files (.vrt format)
│   ├── output_data/                  # Processed output JSON files
│   └── Semantic-search-app/         # Docker setup
│       ├── docker-compose.yml
│       ├── Dockerfile.backend
│       └── Dockerfile.frontend
│
├── frontend/                         # Frontend React application
│   ├── src/
│   │   ├── App.tsx                  # Main app component
│   │   ├── main.tsx                 # Entry point
│   │   ├── components/              # React components
│   │   ├── contexts/                # Context providers
│   │   ├── App.css                  # Global styles
│   │   └── index.css                # Base styles
│   ├── public/                       # Static assets
│   ├── package.json                 # Dependencies
│   ├── vite.config.ts               # Vite configuration
│   ├── tailwind.config.ts           # Tailwind configuration
│   └── tsconfig.json                # TypeScript configuration
│
├── docs/                             # Documentation
│   └── NeuroClip_UseCase_Description.md
│
├── vercel.json                       # Vercel deployment config
└── SETUP_GUIDE.md                   # This file


---

## ▶ Running the Application

### Option 1: Run Backend and Frontend Separately

#### Terminal 1: Start Backend Server

powershell
cd backend
npm start
# or for development with auto-reload
npm run dev


*Expected Output:*

🚀 Server running on http://localhost:5000
Connected to Weaviate at http://localhost:8080


#### Terminal 2: Start Frontend Development Server

powershell
cd frontend
npm run dev
# or using bun
bun run dev


*Expected Output:*

VITE v5.x.x  ready in XXX ms

➜  Local:   http://localhost:5173/
➜  press h + enter to show help


### Option 2: Run with Docker

Navigate to backend/Semantic-search-app/ and use the provided Docker setup:

powershell
cd backend/Semantic-search-app
docker-compose up -d


### Option 3: Run with Python Scripts

For data processing and analysis:

powershell
cd backend

# Populate Weaviate with video data
python -m jupyter notebook weaviate_data_populate.ipynb

# Test transcription
python test_transcribe.py

# Extract data from VRT files
python ../src/extract_data_from_vrt.py


---

## 🔌 API Endpoints

### Backend API (Port 5000)


POST   /api/transcribe          - Transcribe video
GET    /api/videos              - List all videos
GET    /api/videos/:id          - Get video details
POST   /api/search              - Semantic search
POST   /api/videos/:id/blur     - Blur video sections
POST   /api/videos/:id/compress - Compress video


Test endpoints using:

powershell
# Example search query
curl -X POST http://localhost:5000/api/search `
  -H "Content-Type: application/json" `
  -d '{
    "text_query": "Ted Cruz scores a victory",
    "image_query": "group of people"
  }'


---

## 🔐 Environment Configuration

### Backend Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| ASSEMBLYAI_API_KEY | AssemblyAI API key | aai_xxxxxxxxxxxx |
| WEAVIATE_URL | Weaviate instance URL | http://localhost:8080 |
| WEAVIATE_API_KEY | Weaviate API key | your-weaviate-key |
| PORT | Server port | 5000 |
| NODE_ENV | Environment | development or production |
| TEMP_DIR | Temporary files directory | ./temp |
| OUTPUT_DIR | Output data directory | ./output_data |

### Frontend Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| VITE_SUPABASE_URL | Supabase project URL | https://xxxxx.supabase.co |
| VITE_SUPABASE_ANON_KEY | Supabase anonymous key | eyJhbGc... |
| VITE_API_URL | Backend API URL | http://localhost:5000 |
| VITE_APP_NAME | Application name | NeuroClip |

---

## 🛠 Development Commands

### Backend Commands

powershell
# Install dependencies
npm install

# Start server
npm start

# Start with nodemon (auto-restart on changes)
npm run dev

# Run tests
npm test

# Run Python scripts
python script_name.py

# Install Python dependencies
pip install -r requirements.txt


### Frontend Commands

powershell
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Build for development
npm run build:dev

# Preview production build
npm run preview

# Run linter
npm run lint

# Start with bun (faster)
bun run dev


---

## 📊 Data Processing Workflow

### 1. Extract Data from VRT Files

powershell
cd backend/src
python extract_data_from_vrt.py


Outputs: JSON files with text and frame descriptions

### 2. Generate Frame Descriptions

powershell
python frame_desc_all.py


Generates descriptions for all frames in videos

### 3. Populate Weaviate Database

powershell
# Using Jupyter notebook
jupyter notebook weaviate_data_populate.ipynb

# Or directly with Python
python ../insert_video_to_weaviate.py


### 4. Test Search Functionality

powershell
python test_transcribe.py


---

## 🐛 Troubleshooting

### Backend Issues

#### 1. Python Dependencies Installation Fails

powershell
# Try upgrading pip first
python -m pip install --upgrade pip

# Then install requirements
pip install -r requirements.txt

# For GPU issues, install CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


#### 2. Weaviate Connection Error

powershell
# Check if Weaviate is running
# Using Docker
docker ps | grep weaviate

# Using direct URL
curl http://localhost:8080/v1/meta

# If not running, start it
docker run -d -p 8080:8080 semitechnologies/weaviate:latest


#### 3. AssemblyAI API Key Error

- Verify API key in .env file
- Check if API key is active in AssemblyAI dashboard
- Ensure no typos or extra spaces

powershell
# Test API key
python -c "from assemblyai import AssemblyAI; AssemblyAI(api_key='your_key')"


### Frontend Issues

#### 1. Dependencies Installation Fails

powershell
# Clear node_modules and reinstall
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json
npm install


#### 2. Cannot Connect to Backend API

powershell
# Check if backend is running
curl http://localhost:5000/health

# Verify VITE_API_URL in .env.local
# Should be: http://localhost:5000


#### 3. CORS Errors

Ensure backend has CORS enabled in server.js:

javascript
app.use(cors({
  origin: "http://localhost:5173",
  credentials: true
}));


#### 4. Port Already in Use

powershell
# Check what's using port 5000 (backend)
Get-Process -Id (Get-NetTCPConnection -LocalPort 5000).OwningProcess

# Or use different port
$env:PORT=5001

# For frontend (port 5173)
npm run dev -- --port 5174


### Common Solutions

powershell
# Clear all caches
npm cache clean --force

# Reinstall everything
Remove-Item -Recurse -Force node_modules
npm install

# Check versions
node --version
npm --version
python --version

# Update npm
npm install -g npm@latest

# Check disk space
Get-Volume

# Restart services
docker restart <container_id>


---

## 📝 Data Format

### VRT File Structure

The input files use the .vrt format containing:
- Video metadata
- Transcriptions
- Timestamps
- Linguistic features (verbs, etc.)

### Output JSON Format

json
{
  "sentence": "We're going to have a tremendous victory.",
  "starttime": "593.83",
  "endtime": "596.14",
  "verbs": [
    {
      "vword": "going",
      "vstart": "593.83",
      "vend": "593.83",
      "vpos": "VBG"
    }
  ],
  "frame_data": [
    "a man in a suit and red tie.",
    "a picture of donald trump with his mouth wide open."
  ]
}


---

## 🚀 Deployment

### Deploy Frontend to Vercel

The project includes vercel.json configuration:

powershell
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel

# Deploy to production
vercel --prod


### Deploy Backend

powershell
# Using Heroku
heroku create neuroclip-backend
git push heroku main

# Using Railway
railway up

# Using Docker
docker build -t neuroclip-backend .
docker run -p 5000:5000 neuroclip-backend


---

## 📚 Additional Resources

- *Backend Documentation*: See backend/README.md
- *Frontend Documentation*: See frontend/README.md
- *Project Blog*: [NeuroClip Notion Blog](https://dhruv-kunjadiya.notion.site/)
- *Weaviate Docs*: https://weaviate.io/developers/weaviate
- *Supabase Docs*: https://supabase.com/docs
- *React Docs*: https://react.dev

---

## ✅ Quick Start Checklist

- [ ] Install Node.js v18+
- [ ] Install Python 3.8+
- [ ] Clone repository
- [ ] Obtain API keys (AssemblyAI, Supabase)
- [ ] Set up backend .env file
- [ ] Install backend dependencies: pip install -r requirements.txt && npm install
- [ ] Set up frontend .env.local file
- [ ] Install frontend dependencies: npm install or bun install
- [ ] Start Weaviate (Docker or local)
- [ ] Start backend: npm start
- [ ] Start frontend: npm run dev
- [ ] Access app at http://localhost:5173

---

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review error logs in console
3. Check API status and logs
4. Consult project documentation
5. Open an issue on GitHub

---

*Last Updated*: December 2025
*Version*: 1.0.0