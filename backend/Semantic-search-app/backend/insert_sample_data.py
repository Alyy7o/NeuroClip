from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os
import shutil
import uuid
import assemblyai as aai
import requests
from pathlib import Path
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Try to load .env from multiple locations
env_paths = [
    Path(__file__).resolve().parent.parent / ".env",  # Parent directory
    Path(__file__).resolve().parent / ".env",        # Same directory
    Path.cwd() / ".env",                             # Current working directory
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded .env from: {env_path}")
        break
else:
    print("No .env file found in any of the expected locations")

app = FastAPI()

# Lazily initialize the embedding model
model = None

def get_model():
    global model
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise HTTPException(status_code=500, detail="sentence-transformers not installed; search endpoint unavailable")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Configure AssemblyAI
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
if not ASSEMBLYAI_API_KEY:
    raise ValueError("ASSEMBLYAI_API_KEY not found in environment variables")

aai.settings.api_key = ASSEMBLYAI_API_KEY
ASSEMBLYAI_BASE_URL = os.getenv("ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com/v2").rstrip("/")
ASSEMBLYAI_API_ROOT = ASSEMBLYAI_BASE_URL
if ASSEMBLYAI_API_ROOT.endswith("/v2"):
    ASSEMBLYAI_API_ROOT = ASSEMBLYAI_API_ROOT[:-3]
ASSEMBLYAI_IGNORE_PROXIES = os.getenv("ASSEMBLYAI_IGNORE_PROXIES", "true").lower() in ("1", "true", "yes")

def get_requests_session():
    s = requests.Session()
    if ASSEMBLYAI_IGNORE_PROXIES:
        s.trust_env = False
        s.proxies = {}
    return s

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8030",
        "http://localhost:8080",
        "http://localhost:8081",
    ], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "semantic-backend", "port": 8040}

@app.get("/debug-env")
def debug_env():
    return {
        "ASSEMBLYAI_API_KEY_set": bool(ASSEMBLYAI_API_KEY),
        "key_length": len(ASSEMBLYAI_API_KEY),
        "key_preview": ASSEMBLYAI_API_KEY[:4] + "****" if ASSEMBLYAI_API_KEY else None,
        "SUPABASE_URL_set": bool(SUPABASE_URL),
        "SUPABASE_KEY_set": bool(SUPABASE_SERVICE_ROLE_KEY)
    }

@app.get("/search")
def search(text_desc: str = "", video_desc: str = "", n_records: int = 10, min_distance: float = 0.3):
    print("Text length: ", len(text_desc))
    print("Video length: ", len(video_desc))
    print("text_desc: ", text_desc)
    print("video_desc: ", video_desc)

    if len(text_desc) != 0 and len(video_desc) != 0:
        combined_text = f"In the video you can hear: {text_desc} In the video you can see: {video_desc}"
        query_text = combined_text
    elif len(text_desc) != 0:
        query_text = text_desc
    else:
        query_text = video_desc

    query_embedding = get_model().encode(query_text).tolist()
    similarity_threshold = max(0.0, min(1.0, 1.0 - min_distance))

    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY missing)")
    try:
        rpc = supabase.rpc(
            "search_videos",
            {
                "query_embedding": query_embedding,
                "similarity_threshold": similarity_threshold,
                "match_count": n_records,
            },
        ).execute()

        return {"results": rpc.data or [], "count": len(rpc.data or [])}
    except Exception as e:
        print("Supabase search RPC error:", e)
        return {"error": str(e)}

# Helper function to run blocking code in a thread pool
def run_in_threadpool(func):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        return loop.run_in_executor(pool, func)

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    # Prepare directories
    try:
        base_dir = Path(os.getenv("APP_BASE_DIR") or Path(__file__).resolve().parent)
        uploads_dir = base_dir / "uploads"
        srt_dir = base_dir / "srt"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        srt_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare upload directories: {e}")

    # Save uploaded file
    safe_name = Path(file.filename or "uploaded.bin").name
    job_id = str(uuid.uuid4())
    saved_path = uploads_dir / f"{job_id}_{safe_name}"
    try:
        with saved_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    # Validate file extension
    allowed_ext = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".webm", ".ogg", ".flac"}
    ext = saved_path.suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported media type '{ext}'. Please upload audio/video (e.g., .mp3, .wav, .mp4)")

    # Check AssemblyAI API key
    if not aai.settings.api_key:
        raise HTTPException(status_code=400, detail="ASSEMBLYAI_API_KEY not configured")
    
    headers = {"authorization": aai.settings.api_key}
    upload_url = f"{ASSEMBLYAI_BASE_URL}/upload"
    session = get_requests_session()
    
    # Upload file to AssemblyAI
    try:
        with saved_path.open("rb") as f:
            def gen():
                while True:
                    chunk = f.read(5 * 1024 * 1024)  # 5MB chunks
                    if not chunk:
                        break
                    yield chunk
            up_resp = session.post(upload_url, headers=headers, data=gen())
            up_resp.raise_for_status()
            response_data = up_resp.json()
            audio_url = response_data.get("upload_url") or response_data.get("url")
            if not audio_url:
                raise HTTPException(status_code=502, detail="Upload did not return an audio URL")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upload failed: {e}")

    # Create transcript
    try:
        create_resp = session.post(
            f"{ASSEMBLYAI_BASE_URL}/transcripts",
            headers={**headers, "content-type": "application/json"},
            json={"audio_url": audio_url},
            timeout=30,
        )

        transcript_id = None
        used_sdk_fallback = False
        transcript_obj = None

        try:
            create_resp.raise_for_status()
        except requests.HTTPError:
            # If we get a 404, try SDK fallback
            if create_resp is not None and create_resp.status_code == 404:
                try:
                    # Set SDK base URL
                    aai.settings.base_url = ASSEMBLYAI_API_ROOT
                    
                    # Create a config object
                    config = aai.TranscriptionConfig(
                        language_code="en",  # Set your desired language
                        punctuate=True,
                        format_text=True
                    )
                    
                    # Use SDK for transcription
                    transcriber = aai.Transcriber()
                    transcript_obj = transcriber.transcribe(str(saved_path), config=config)
                    
                    # Check if transcription was successful
                    if transcript_obj.status == aai.TranscriptStatus.error:
                        raise HTTPException(status_code=502, detail=f"SDK transcription failed: {transcript_obj.error}")
                    
                    transcript_id = transcript_obj.id
                    if not transcript_id:
                        raise HTTPException(status_code=502, detail="SDK fallback did not return transcript id")
                    used_sdk_fallback = True
                except Exception as sdk_err:
                    body = None
                    try:
                        body = create_resp.text
                    except Exception:
                        pass
                    raise HTTPException(status_code=502, detail=f"Transcript creation failed (REST 404) and SDK fallback errored: {sdk_err}{f' | response: {body}' if body else ''}")
            else:
                # Non-404 error; propagate
                raise

        # If REST was successful, parse id from JSON
        if transcript_id is None:
            try:
                transcript_id = create_resp.json().get("id")
            except Exception as parse_err:
                raise HTTPException(status_code=502, detail=f"Transcript creation JSON parse failed: {parse_err} | response: {create_resp.text}")

        if not transcript_id:
            raise HTTPException(status_code=502, detail=f"Transcript creation did not return an id. Response: {create_resp.text}")
    except HTTPException:
        raise
    except Exception as e:
        body = None
        try:
            body = create_resp.text
        except Exception:
            pass
        raise HTTPException(status_code=502, detail=f"Transcript creation failed: {e}{f' | response: {body}' if body else ''}")

    # If we used SDK fallback and have a completed transcript_obj, export SRT via SDK and return immediately
    if used_sdk_fallback and transcript_obj is not None:
        try:
            srt_path = srt_dir / (saved_path.stem + ".srt")
            srt_text = transcript_obj.export_subtitles_srt()
            srt_path.write_text(srt_text, encoding="utf-8")
            return {
                "message": "Upload and transcription complete (SDK)",
                "job_id": job_id,
                "transcript_id": transcript_id,
                "video_path": str(saved_path),
                "srt_path": str(srt_path),
            }
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"SDK SRT export failed: {e}")

    # Poll status until completed (REST)
    status_url = f"{ASSEMBLYAI_BASE_URL}/transcripts/{transcript_id}"
    max_wait_env = os.getenv("ASSEMBLYAI_MAX_WAIT", "0")
    try:
        max_wait_seconds = int(max_wait_env)
    except Exception:
        max_wait_seconds = 0
    poll_interval_seconds = int(os.getenv("ASSEMBLYAI_POLL_INTERVAL", "3"))
    start_time = time.time()
    last_status = None
    consecutive_poll_errors = 0
    
    while True:
        try:
            resp = session.get(status_url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            last_status = data.get("status")
            if last_status == "completed":
                break
            if last_status == "error":
                err = data.get("error", "Unknown transcription error")
                raise HTTPException(status_code=502, detail=f"Transcription error: {err}")
        except HTTPException:
            raise
        except Exception as e:
            last_status = f"poll_error: {e}"
            consecutive_poll_errors += 1
            if consecutive_poll_errors >= 5:
                raise HTTPException(status_code=502, detail=f"Transcript poll repeatedly failed ({consecutive_poll_errors} errors). Last: {last_status}")
        
        # Check timeout
        if max_wait_seconds > 0 and (time.time() - start_time > max_wait_seconds):
            raise HTTPException(status_code=504, detail=f"Transcript still processing after {max_wait_seconds}s (last status: {last_status})")
        
        # Wait before next poll
        await asyncio.sleep(poll_interval_seconds)

    # Export SRT to file
    srt_path = srt_dir / (saved_path.stem + ".srt")
    try:
        url = f"{ASSEMBLYAI_BASE_URL}/transcripts/{transcript_id}/srt"
        r = session.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        srt_path.write_text(r.text, encoding="utf-8")
    except Exception:
        raise HTTPException(status_code=502, detail="SRT export failed after transcript completion")

    return {
        "message": "Upload and transcription complete",
        "job_id": job_id,
        "transcript_id": transcript_id,
        "video_path": str(saved_path),
        "srt_path": str(srt_path),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8040)