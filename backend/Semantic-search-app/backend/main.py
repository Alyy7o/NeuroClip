import os
import sys
import shutil
import uuid
import json
import re
import time
import requests
import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
import assemblyai as aai
from pathlib import Path
from typing import List, Optional
import torch
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- Path Definitions ---
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]

# --- App Initialization ---
API_BUILD_ID = "2026-05-blur-v1"
app = FastAPI()


def _registered_paths() -> list:
    paths = []
    for route in app.routes:
        p = getattr(route, "path", None)
        if p:
            paths.append(p)
    return sorted(set(paths))

# --- Module Imports with Path Fixes ---
try:
    from assemblyai_utils import generate_transcript_from_video
except ImportError:
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    from assemblyai_utils import generate_transcript_from_video

try:
    from ocr_utils import load_ocr_model, extract_high_value_frames, run_ocr_on_frames
except ImportError:
    if str(BASE_DIR) not in sys.path:
        sys.path.append(str(BASE_DIR))
    from ocr_utils import load_ocr_model, extract_high_value_frames, run_ocr_on_frames

try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# --- Environment Variables ---
# Priority 1: Kaggle Secrets — using the exact API pattern from Kaggle
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    # Load each secret and inject into os.environ immediately
    os.environ["ASSEMBLYAI_API_KEY"]      = user_secrets.get_secret("ASSEMBLYAI_API_KEY")
    os.environ["GOOGLE_API_KEY"]          = user_secrets.get_secret("GOOGLE_API_KEY")
    os.environ["GROQ_API_KEY"]            = user_secrets.get_secret("GROQ_API_KEY")
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = user_secrets.get_secret("SUPABASE_SERVICE_ROLE_KEY")
    os.environ["SUPABASE_URL"]            = user_secrets.get_secret("SUPABASE_URL")
    os.environ["VITE_SUPABASE_ANON_KEY"]  = user_secrets.get_secret("VITE_SUPABASE_ANON_KEY")
    # Copy SUPABASE_SERVICE_ROLE as alias (some code references this name)
    os.environ.setdefault("SUPABASE_SERVICE_ROLE", os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""))
    os.environ.setdefault("VITE_SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
    print("[secrets] All Kaggle secrets loaded into os.environ ✓")
except Exception as _kaggle_err:
    print(f"[secrets] Kaggle secrets not available ({_kaggle_err}) — falling back to .env files")
    # Priority 2: .env files (local dev / non-Kaggle servers)
    _env_candidates = [
        BASE_DIR.parent / ".env",          # Semantic-search-app/.env
        REPO_ROOT / ".env",                # same level alias
        BASE_DIR.parent.parent / ".env",   # backend/.env
        BASE_DIR / ".env",                 # backend/Semantic-search-app/backend/.env
        Path("/kaggle/working") / ".env",  # Kaggle fall-through
    ]
    try:
        from dotenv import load_dotenv
        for p in _env_candidates:
            if p.exists():
                load_dotenv(str(p), override=True)
                print(f"[env] Loaded: {p}")
    except ImportError:
        for p in _env_candidates:
            try:
                if p.exists():
                    for _line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                        _s = _line.strip()
                        if not _s or _s.startswith("#") or "=" not in _s:
                            continue
                        _k, _v = _s.split("=", 1)
                        _k = _k.strip().strip('"').strip("'")
                        _v = _v.strip().strip('"').strip("'")
                        if _k and _v:
                            os.environ[_k] = _v
            except Exception:
                continue

# Startup diagnostic
_aai_key = os.getenv("ASSEMBLYAI_API_KEY", "")
print(f"[startup] BASE_DIR={BASE_DIR}")
print(f"[startup] REPO_ROOT={REPO_ROOT}")
print(f"[startup] ASSEMBLYAI_API_KEY={'SET (len='+str(len(_aai_key))+')' if _aai_key.strip() else 'NOT SET!'}")
print(f"[startup] SUPABASE_URL={'SET' if os.getenv('SUPABASE_URL') else 'not set'}")
print(f"[startup] GOOGLE_API_KEY={'SET' if os.getenv('GOOGLE_API_KEY') else 'not set'}")
print(f"[startup] GROQ_API_KEY={'SET' if os.getenv('GROQ_API_KEY') else 'NOT SET — add to Kaggle secrets!'}")
print(f"[startup] LLM Provider Priority: {'Groq → Gemini' if os.getenv('GROQ_API_KEY') else 'Gemini only'}")



# Ensure required directories exist for StaticFiles
(REPO_ROOT / "uploads").mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "output_data").mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "input_files").mkdir(parents=True, exist_ok=True)

# Lazily initialize the embedding model to avoid heavy startup
model = None
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence").lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
# Ensure model caches persist so weights are downloaded once
DEFAULT_CACHE = str(REPO_ROOT / ".cache")
os.environ.setdefault("HF_HOME", DEFAULT_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", DEFAULT_CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", DEFAULT_CACHE)

def get_model():
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if EMBEDDING_BACKEND == "sentence":
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
            except Exception:
                raise HTTPException(status_code=500, detail="sentence-transformers unavailable; set EMBEDDING_BACKEND=openclip")
        elif EMBEDDING_BACKEND == "openclip":
            try:
                import open_clip
                # NOTE: Do NOT import torch here — it shadows the global import
                # and causes UnboundLocalError in the sentence-transformers branch
                class _OpenClipEncoder:
                    def __init__(self, device):
                        self.device = device
                        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                            'ViT-B-32', pretrained='laion2b_s34b_b79k'
                        )
                        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
                        self.model.to(self.device)
                        self.model.eval()
                    def encode(self, texts):
                        with torch.no_grad():
                            tokens = self.tokenizer(texts).to(self.device)
                            feats = self.model.encode_text(tokens)
                            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-9)
                            return feats.cpu().numpy().tolist()
                model = _OpenClipEncoder(device)
            except Exception:
                raise HTTPException(status_code=500, detail="open-clip-torch unavailable; set EMBEDDING_BACKEND=sentence")
        else:
            raise HTTPException(status_code=500, detail="Invalid EMBEDDING_BACKEND; use 'sentence' or 'openclip'")
    return model

# Initialize Supabase client (use Service Role key for server-side inserts/RPC)
SUPABASE_URL = os.getenv("SUPABASE_URL", os.getenv("VITE_SUPABASE_URL", ""))
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_SERVICE_ROLE", ""))
# Initialize Supabase only if configured; keep optional for routes that don't need it
supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY and create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    except Exception:
        supabase = None

# Configure AssemblyAI
api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
if api_key:
    aai.settings.api_key = api_key
# REST base includes "/v2"; SDK expects base WITHOUT "/v2" and appends it internally.
ASSEMBLYAI_BASE_URL = os.getenv("ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com/v2").rstrip("/")
ASSEMBLYAI_API_ROOT = ASSEMBLYAI_BASE_URL
if ASSEMBLYAI_API_ROOT.endswith("/v2"):
    ASSEMBLYAI_API_ROOT = ASSEMBLYAI_API_ROOT[:-3]  # strip trailing "/v2" for SDK usage
ASSEMBLYAI_IGNORE_PROXIES = os.getenv("ASSEMBLYAI_IGNORE_PROXIES", "true").lower() in ("1", "true", "yes")

def get_requests_session():
    s = requests.Session()
    # Avoid using environment proxies that may rewrite or block requests
    if ASSEMBLYAI_IGNORE_PROXIES:
        s.trust_env = False
        s.proxies = {}
    return s

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (ngrok, localhost, deployed frontend)
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(REPO_ROOT / "output_data")),
    name="static",
)

app.mount(
    "/uploads",
    StaticFiles(directory=str(REPO_ROOT / "uploads")),
    name="uploads",
)

@app.get("/download")
def download_file(path: str):
    """
    Force file download (Content-Disposition: attachment).
    'path' is relative to repo root (e.g. 'backend/output_data/clips/...' or 'backend/uploads/...')
    or we can support specific sub-roots.
    This implementation supports two roots: 'uploads' and 'output_data'.
    path param expected: 'uploads/filename.mp4' or 'clips/jobid/clip.mp4'
    """
    # Defensive logic to prevent arbitrary file reading
    # We resolve paths relative to the backend parent directory (the repo root context we used elsewhere)
    
    # Base roots
    base_uploads = REPO_ROOT / "uploads"
    base_output = REPO_ROOT / "output_data"
    
    target_file = None
    
    # Naive routing based on prefix
    if path.startswith("uploads/"):
        # e.g. uploads/uuid.mp4
        filename = path.replace("uploads/", "")
        target_file = base_uploads / filename
    elif path.startswith("clips/"):
        # e.g. clips/job_id/clip_01.mp4 
        target_file = base_output / path
    else:
        # Fallback: check if it matches a direct file in output_data (like srt)
        potential = base_output / path
        if potential.exists() and potential.is_file():
             target_file = potential
    
    if not target_file or not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        path=target_file, 
        filename=target_file.name, 
        media_type='application/octet-stream'
    )

@app.get("/serve-clip")
def serve_clip(path: str):
    """
    Serve a video clip inline for playback.
    Unlike /static which gets blocked by ngrok's browser interstitial,
    this endpoint serves files directly with the correct video/mp4 MIME type.
    
    path: relative path under output_data, e.g. 'clips/UUID/clip_01.mp4'
    """
    base_output = REPO_ROOT / "output_data"
    
    # Security: prevent path traversal
    safe_path = Path(path).as_posix()
    if ".." in safe_path:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    target_file = base_output / safe_path
    
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"Clip not found: {safe_path}")
    
    # Determine MIME type
    suffix = target_file.suffix.lower()
    mime_map = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}
    media_type = mime_map.get(suffix, "video/mp4")
    
    return FileResponse(
        path=str(target_file),
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600",
        },
    )


# --- SRT to JSON utilities (project format) ---
TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})$")

def _hms_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def _normalize_text(lines):
    text = " ".join(line.strip() for line in lines if line is not None and line.strip())
    return re.sub(r"\s+", " ", text).strip()

def parse_srt_blocks(srt_content: str):
    sentences = []
    blocks = re.split(r"\r?\n\r?\n+", srt_content.strip(), flags=re.MULTILINE)
    for block in blocks:
        lines = [l for l in block.splitlines()]
        if not lines:
            continue
        if len(lines) >= 2 and TIME_RE.match(lines[1]):
            time_line_idx = 1
            text_lines = lines[2:]
        elif TIME_RE.match(lines[0]):
            time_line_idx = 0
            text_lines = lines[1:]
        else:
            continue
        m = TIME_RE.match(lines[time_line_idx])
        if not m:
            continue
        start = _hms_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _hms_to_seconds(m.group(5), m.group(6), m.group(7), m.group(8))
        text = _normalize_text(text_lines)
        if not text:
            continue
        sentences.append({
            "sentence": text,
            "starttime": f"{start:.2f}",
            "endtime": f"{end:.2f}",
            "verbs": []
        })
    return sentences

def build_metadata(base: Optional[dict], srt_path: str, video_path: Optional[str] = None) -> dict:
    from datetime import datetime
    now = datetime.now()
    defaults = {
        "text_id": f"t__{uuid.uuid4().hex[:8]}_{uuid.uuid4().hex[:4]}_{uuid.uuid4().hex[:4]}",
        "collection": "N/A",
        "file": video_path or srt_path,
        "date": now.strftime("%Y-%m-%d"),
        "year": now.strftime("%Y"),
        "month": now.strftime("%m"),
        "day": now.strftime("%d"),
        "time": now.strftime("%H%M"),
        "duration": "N/A",
        "country": "N/A",
        "channel": "N/A",
        "title": Path(video_path).stem if video_path else Path(srt_path).stem,
        "video_resolution": "N/A",
        "video_resolution_original": "N/A",
        "language": "ENG",
        "recording_location": "N/A",
        "original_broadcast_date": "N/A",
        "original_broadcast_time": "N/A",
        "original_broadcast_timezone": "N/A",
        "local_broadcast_date": now.strftime("%Y-%m-%d"),
        "local_broadcast_time": now.strftime("%H:%M"),
        "local_broadcast_timezone": "N/A",
    }
    if base:
        defaults.update(base)
    return defaults


class TranscribeURLRequest(BaseModel):
    url: str
    language_code: Optional[str] = None
    save_srt: bool = False
    srt_filename: Optional[str] = None


@app.post("/assemblyai/transcribe-url")
def transcribe_url(payload: TranscribeURLRequest):
    out_path = None
    if payload.save_srt:
        out_dir = REPO_ROOT / "output_data"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = payload.srt_filename or "AssemblyAI_transcript.srt"
        out_path = out_dir / fname

    result = generate_transcript_from_video(
        source=str(payload.url),
        output_srt_path=str(out_path) if out_path else None,
        language_code=payload.language_code,
    )

    return {
        "id": result.get("id", ""),
        "status": result.get("status", ""),
        "text": result.get("text", ""),
        "srt_path": str(result.get("srt_path")) if result.get("srt_path") else None,
    }


class ConvertSrtRequest(BaseModel):
    srt_path: Optional[str] = None
    srt_text: Optional[str] = None
    video_path: Optional[str] = None
    metadata: Optional[dict] = None


@app.post("/srt/convert")
def convert_srt(payload: ConvertSrtRequest):
    """Convert an SRT file (from AssemblyAI or similar) into project JSON format.

    - Input: `srt_path` on disk (e.g., from /upload-video response)
    - Optional: `video_path` to populate metadata.file/title
    - Optional: `metadata` dict to override default metadata fields
    """
    if payload.srt_text:
        srt_content = payload.srt_text
        srt_path_str = payload.srt_path or ""
    else:
        if not payload.srt_path:
            raise HTTPException(status_code=400, detail="Provide either srt_text or srt_path")
        srt_path = Path(payload.srt_path)
        if not srt_path.exists():
            raise HTTPException(status_code=404, detail=f"SRT file not found: {srt_path}")
        try:
            srt_content = srt_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read SRT: {e}")
        srt_path_str = str(srt_path)

    sentences = parse_srt_blocks(srt_content)
    # Compute duration as last endtime (string seconds) if available
    duration_seconds = sentences[-1]["endtime"] if sentences else "0.00"
    meta = build_metadata(payload.metadata or {}, srt_path_str, payload.video_path)
    if duration_seconds and meta.get("duration") in (None, "N/A"):
        meta["duration"] = duration_seconds

    data = {
        "metadata": meta,
        "sentences": sentences,
    }
    return data


@app.post("/assemblyai/transcribe-file")
async def transcribe_file(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form(None),
    save_srt: bool = Form(False),
    srt_filename: Optional[str] = Form(None),
):
    temp_dir = REPO_ROOT / "input_files"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"transcribe_{uuid.uuid4()}_{file.filename}"

    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    out_path = None
    if save_srt:
        out_dir = REPO_ROOT / "output_data"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = srt_filename or (Path(file.filename).stem + ".srt")
        out_path = out_dir / fname

    try:
        result = generate_transcript_from_video(
            source=str(temp_path),
            output_srt_path=str(out_path) if out_path else None,
            language_code=language_code,
        )
    except ValueError as e:
        # generate_transcript_from_video raises ValueError when API key is missing
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")
    finally:
        # Cleanup uploaded temp file
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "id": result.get("id", ""),
        "status": result.get("status", ""),
        "text": result.get("text", ""),
        "srt_path": str(result.get("srt_path")) if result.get("srt_path") else None,
    }

@app.on_event("startup")
def _startup_route_audit():
    has_anonymize = "/anonymize-video" in _registered_paths()
    if not has_anonymize:
        print("[startup] WARNING: POST /anonymize-video is NOT registered — blur UI will 404")


@app.get("/health")
def health():
    """Simple health check to verify service is reachable."""
    blur_loaded = False
    cuda_available = False
    paths = _registered_paths()
    has_anonymize = "/anonymize-video" in paths
    try:
        from blur_service import blur_model_loaded
        blur_loaded = blur_model_loaded()
    except Exception:
        pass
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        pass
    return {
        "status": "ok",
        "service": "semantic-backend",
        "port": 8040,
        "build_id": API_BUILD_ID,
        "cuda_available": cuda_available,
        "blur_model_loaded": blur_loaded,
        "has_anonymize_endpoint": has_anonymize,
        "has_compress_endpoint": "/compress-video" in paths,
    }

@app.get("/search")
def search(text_desc: str = "", video_desc: str = "", n_records: int = 10, min_distance: float = 0.3):
    print ("Text length: ", len(text_desc))
    print ("Video length: ", len(video_desc))
    print ("text_desc: ", text_desc)
    print ("video_desc: ", video_desc)

    # Build the text we want to embed
    if len(text_desc) != 0 and len(video_desc) != 0:
        combined_text = f"In the video you can hear: {text_desc} In the video you can see: {video_desc}"
        query_text = combined_text
    elif len(text_desc) != 0:
        query_text = text_desc
    else:
        query_text = video_desc

    # Generate embedding (dimension ~384 for all-MiniLM-L6-v2)
    query_embedding = get_model().encode(query_text).tolist()

    # Convert Weaviate 'distance' threshold to similarity threshold loosely
    # Weaviate distance ~ lower is better; pgvector similarity = 1 - cosine_distance
    similarity_threshold = max(0.0, min(1.0, 1.0 - min_distance))

    # Call Supabase RPC for semantic search (requires SQL function `search_videos`)
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

        # rpc.data will contain rows from the function
        return {"results": rpc.data or [], "count": len(rpc.data or [])}
    except Exception as e:
        print("Supabase search RPC error:", e)
        return {"error": str(e)}



def process_video_workflow(
    saved_path: Path, 
    job_id: str, 
    user_id: Optional[str] = None, 
    query: Optional[str] = None,
    original_filename: str = "video",
    provided_srt_path: Optional[Path] = None,
    input_source: Optional[str] = None
):

    """
    Common workflow with PARALLEL PROCESSING:
    1. OCR (frame extraction + text recognition)   ← runs in parallel
    2. Transcribe (AssemblyAI or provided SRT)      ← runs in parallel
    3. Merge OCR into transcript
    4. Generate Embeddings
    5. Save to Database (Supabase)
    """
    pipeline_start = time.time()
    output_dir = REPO_ROOT / "output_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    srt_out = output_dir / f"{saved_path.stem}.srt"

    # =====================================================================
    # PARALLEL TASK 1: OCR (Frame Extraction + Text Recognition)
    # =====================================================================
    def _run_ocr_task():
        """Extract frames from video and run OCR on them. Runs in a thread."""
        t0 = time.time()
        try:
            from ocr_utils import extract_all_video_frames, run_ocr_on_frames as ocr_run
            frames_dir = output_dir / "frames" / job_id
            print(f"[OCR] Starting full-video frame extraction to {frames_dir}...")
            count = extract_all_video_frames(saved_path, frames_dir)
            if count > 0:
                ocr_results = ocr_run(frames_dir)
                elapsed = time.time() - t0
                print(f"[OCR] Completed in {elapsed:.1f}s: {len(ocr_results)} frames contained text")
                return ocr_results
            else:
                print("[OCR] No frames could be extracted from video")
                return []
        except Exception as e:
            print(f"[OCR ERROR] OCR/Frame extraction failed: {e}")
            return []

    # =====================================================================
    # PARALLEL TASK 2: Transcription (AssemblyAI or provided SRT)
    # =====================================================================
    def _run_transcription_task():
        """Transcribe video via AssemblyAI or use provided SRT. Runs in a thread."""
        t0 = time.time()
        result = {
            "id": "",
            "status": "",
            "text": "",
            "srt_path": None,
        }

        # Use provided SRT if available (YouTube captions)
        if provided_srt_path and provided_srt_path.exists():
            try:
                shutil.copy2(provided_srt_path, srt_out)
                srt_content = srt_out.read_text(encoding="utf-8", errors="replace")
                result = {
                    "id": "yt_caption",
                    "status": "completed",
                    "text": srt_content,
                    "srt_path": srt_out
                }
                elapsed = time.time() - t0
                print(f"[TRANSCRIPTION] Used provided SRT captions in {elapsed:.1f}s")
                return result
            except Exception as e:
                print(f"[TRANSCRIPTION] Failed to use provided SRT: {e} — falling back to AssemblyAI")

        # AssemblyAI transcription
        if result.get("status") != "completed":
            try:
                print(f"[TRANSCRIPTION] Starting AssemblyAI transcription...")
                result = generate_transcript_from_video(
                    source=str(saved_path),
                    output_srt_path=str(srt_out),
                    language_code=None,
                )
                elapsed = time.time() - t0
                print(f"[TRANSCRIPTION] AssemblyAI completed in {elapsed:.1f}s")
            except ValueError as e:
                raise e  # Will be caught in main thread
            except Exception as e:
                raise e  # Will be caught in main thread

        return result

    # =====================================================================
    # RUN TASKS IN PARALLEL
    # =====================================================================
    print(f"[PIPELINE] Starting parallel processing for job {job_id}...")
    parallel_start = time.time()

    ocr_text_data = []
    result = {"id": "", "status": "", "text": "", "srt_path": None}
    transcription_error = None

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="neuro") as executor:
        # Submit both tasks simultaneously
        ocr_future = executor.submit(_run_ocr_task)
        transcription_future = executor.submit(_run_transcription_task)

        # Collect results as they complete
        for future in as_completed([ocr_future, transcription_future]):
            try:
                if future == ocr_future:
                    ocr_text_data = future.result()
                elif future == transcription_future:
                    result = future.result()
            except ValueError as e:
                transcription_error = ("value", e)
            except Exception as e:
                transcription_error = ("general", e)

    parallel_elapsed = time.time() - parallel_start
    print(f"[PIPELINE] Parallel phase completed in {parallel_elapsed:.1f}s "
          f"(OCR: {len(ocr_text_data)} text items, Transcription: {result.get('status', 'unknown')})")

    # Handle transcription errors after parallel phase
    if transcription_error:
        err_type, err = transcription_error
        if err_type == "value":
            raise HTTPException(status_code=400, detail=str(err))
        else:
            raise HTTPException(status_code=502, detail=f"Transcription failed: {err}")

    # =====================================================================
    # SEQUENTIAL PHASE: Merge + Embed + Save
    # =====================================================================
    merge_start = time.time()

    # Convert SRT text to project JSON
    srt_text = ""
    if result.get("srt_path"):
        try:
             srt_text = Path(result["srt_path"]).read_text(encoding="utf-8", errors="replace")
        except:
             srt_text = result.get("text", "")
    elif result.get("text"):
        srt_text = result.get("text")
    
    if not srt_text:
         if srt_out.exists():
             srt_text = srt_out.read_text(encoding="utf-8", errors="replace")

    sentences = parse_srt_blocks(srt_text)
    
    # Merge OCR text into sentences — attach to the BEST matching sentence
    if ocr_text_data and sentences:
        for ocr_item in ocr_text_data:
            ts = ocr_item["timestamp"]
            text = ocr_item["text"]
            best_match = None
            best_dist = float('inf')
            for s in sentences:
                s_start = float(s["starttime"])
                s_end = float(s["endtime"])
                if s_start <= ts <= s_end:
                    best_match = s
                    best_dist = 0
                    break
                dist = min(abs(s_start - ts), abs(s_end - ts))
                if dist < 5.0 and dist < best_dist:
                    best_dist = dist
                    best_match = s
            if best_match is not None:
                best_match["sentence"] += f" [On Screen: {text}]"
            else:
                sentences.append({
                    "sentence": f"[Visual Content]: {text}",
                    "starttime": f"{ts:.2f}",
                    "endtime": f"{ts+2.0:.2f}",
                    "verbs": ["visual_ocr"]
                })
        sentences.sort(key=lambda x: float(x["starttime"]))

    merge_elapsed = time.time() - merge_start
    print(f"[PIPELINE] SRT parsing + OCR merge completed in {merge_elapsed:.1f}s ({len(sentences)} sentences)")

    duration_seconds = sentences[-1]["endtime"] if sentences else "0.00"
    meta = build_metadata({}, srt_path=str(saved_path.with_suffix('.srt')), video_path=str(saved_path))
    if duration_seconds and meta.get("duration") in (None, "N/A"):
        meta["duration"] = duration_seconds

    data = {"metadata": meta, "sentences": sentences}
    
    # Generate embeddings
    embed_start = time.time()
    emb = {}
    try:
        vecs = get_model().encode([s["sentence"] for s in sentences])
        emb = {
            "vectors": [list(map(float, v)) for v in vecs],
            "sentences": sentences,
        }
        (output_dir / f"{saved_path.stem}.embeddings.json").write_text(
            json.dumps(emb, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

    embed_elapsed = time.time() - embed_start
    print(f"[PIPELINE] Embedding generation completed in {embed_elapsed:.1f}s")
        
    json_path = output_dir / f"{saved_path.stem}.v4.json"
    try:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write JSON: {e}")

    # Persist to Supabase
    db_start = time.time()
    history_saved = False
    try:
        if supabase is not None:
            if user_id:
                try:
                    supabase.table("user_videos").upsert({
                        "id": job_id,
                        "job_id": job_id,
                        "user_id": user_id,
                        "title": meta.get("title") or original_filename,
                        "original_filename": original_filename,
                        "video_url": str(saved_path),
                        "file_size": int(saved_path.stat().st_size) if saved_path.exists() else 0,
                        "duration": float(meta.get("duration") or 0),
                        "metadata": {
                            "json_path": str(json_path),
                            "srt_path": str(srt_out),
                        },
                        "status": "completed",
                    }).execute()
                except Exception as e:
                    print("user_videos upsert failed:", e)
            
            try:
                avg_vec = None
                transcript_text = " ".join([s.get("sentence", "") for s in sentences])
                transcript_vec = None
                
                if emb.get("vectors"):
                    import numpy as np
                    avg_vec = np.array(emb["vectors"], dtype=float).mean(axis=0).tolist()
                
                try:
                    transcript_vec = list(map(float, get_model().encode([transcript_text])[0])) if transcript_text else None
                except:
                    pass
                    
                ve_row = {
                    "job_id": job_id,
                    "video_url": str(saved_path),
                    "title": meta.get("title") or original_filename,
                    "description": None,
                    "duration": int(float(meta.get("duration") or 0)),
                    "thumbnail_url": None,
                    "transcript": transcript_text or None,
                    "transcript_id": result.get("id", "") or None,
                    "video_embedding": avg_vec,
                    "transcript_embedding": transcript_vec,
                    "audio_url": None,
                    "frames_path": None,
                }
                supabase.table("video_embeddings").insert(ve_row).execute()
            except Exception as e:
                pass

            # Save sentence embeddings
            try:
                emb_rows = []
                for idx, s in enumerate(sentences):
                    emb_rows.append({
                        "job_id": job_id,
                        "sentence_index": idx,
                        "text": s.get("sentence"),
                        "start": float(s.get("starttime", 0)),
                        "end": float(s.get("endtime", 0)),
                        "embedding": [float(x) for x in emb.get("vectors", [[]])[idx]] if emb.get("vectors") else None,
                    })
                if emb_rows:
                    batch = 200
                    for i in range(0, len(emb_rows), batch):
                        try:
                            supabase.table("video_sentence_embeddings").insert(emb_rows[i:i+batch]).execute()
                        except Exception:
                            pass
            except Exception:
                pass

            # Save processing history
            if user_id:
                try:
                    # Ensure user profile exists (avoids FK constraint violation)
                    try:
                        supabase.table("profiles").upsert(
                            {
                                "id": user_id,
                                "email": f"bot_{user_id[:8]}@neuroclip.eval",
                                "full_name": "Eval Bot",
                            },
                            on_conflict="id"
                        ).execute()
                    except Exception as profile_err:
                        print(f"Profile upsert skipped: {profile_err}")

                    history_data = {
                        "user_id": user_id,
                        "video_id": job_id,
                        "module": "summarization",
                        "input_type": "url" if provided_srt_path else "file",
                        "input_url": input_source or original_filename,
                        "query": query or None,
                        "status": "completed",
                    }
                    r = supabase.table("processing_history").insert(history_data).execute()
                    history_saved = bool(getattr(r, "data", None))
                except Exception as e:
                    print("processing_history insert failed:", e)
                    history_saved = False
    except Exception:
        pass

    db_elapsed = time.time() - db_start
    total_elapsed = time.time() - pipeline_start
    print(f"[PIPELINE] Database persistence completed in {db_elapsed:.1f}s")
    print(f"[PIPELINE] ════════════════════════════════════════════")
    print(f"[PIPELINE] TOTAL PROCESSING TIME: {total_elapsed:.1f}s")
    print(f"[PIPELINE]   ├─ Parallel phase (OCR + Transcription): {parallel_elapsed:.1f}s")
    print(f"[PIPELINE]   ├─ Merge + Parse: {merge_elapsed:.1f}s")
    print(f"[PIPELINE]   ├─ Embeddings: {embed_elapsed:.1f}s")
    print(f"[PIPELINE]   └─ Database: {db_elapsed:.1f}s")
    print(f"[PIPELINE] ════════════════════════════════════════════")

    return {
        "message": "Processing complete",
        "job_id": job_id,
        "transcript_id": result.get("id", ""),
        "status": result.get("status", ""),
        "text": result.get("text", ""),
        "video_path": str(saved_path),
        "json_path": str(json_path),
        "srt_path": str(srt_out),
        "data": data,
        "history_saved": history_saved,
        "processing_time": round(total_elapsed, 2),
    }

class UploadUrlRequest(BaseModel):
    url: str
    query: Optional[str] = None
    user_id: Optional[str] = None

@app.post("/upload-via-url")
def upload_via_url(payload: UploadUrlRequest):
    base_dir = Path(os.getenv("APP_BASE_DIR") or Path(__file__).resolve().parent)
    uploads_dir = base_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    # Deduplication: Check if URL was already processed (or is currently processing)
    if payload.user_id:
        try:
            # Find latest completed/processing job for this URL
            existing = supabase.table("processing_history")\
                .select("video_id, status")\
                .eq("user_id", payload.user_id)\
                .eq("input_url", payload.url)\
                .in_("status", ["completed", "processing"])\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if existing.data and len(existing.data) > 0:
                old_job_id = existing.data[0]["video_id"]
                # Verify we actually have the data
                ve = supabase.table("video_embeddings").select("*").eq("job_id", old_job_id).limit(1).execute()
                if ve.data and len(ve.data) > 0:
                    meta = ve.data[0]
                    # Check if files exist
                    old_vid_path = meta.get("video_url")
                    old_json_path = (meta.get("metadata") or {}).get("json_path")
                    if old_vid_path and Path(old_vid_path).exists() and old_json_path and Path(old_json_path).exists():
                         # Found! Reuse.
                         print(f"Reusing existing job {old_job_id} for URL {payload.url}")
                         
                         # Insert NEW history entry for this NEW query
                         try:
                             # Ensure profile exists
                             try:
                                 supabase.table("profiles").upsert(
                                     {"id": payload.user_id, "email": f"bot_{payload.user_id[:8]}@neuroclip.eval", "full_name": "Eval Bot"},
                                     on_conflict="id"
                                 ).execute()
                             except Exception:
                                 pass
                             history_data = {
                                "user_id": payload.user_id,
                                "video_id": old_job_id,
                                "module": "summarization",
                                "input_type": "url",
                                "input_url": payload.url,
                                "query": payload.query or None,
                                "status": "completed",
                             }
                             supabase.table("processing_history").insert(history_data).execute()
                         except Exception:
                             pass
                             
                         return {
                            "message": "Processing complete (cached)",
                            "job_id": old_job_id,
                            "video_path": old_vid_path,
                            "json_path": old_json_path,
                            "srt_path": (meta.get("metadata") or {}).get("srt_path"),
                            "data": json.loads(Path(old_json_path).read_text(encoding="utf-8")) if Path(old_json_path).exists() else {},
                            "history_saved": True
                        }
        except Exception as e:
            print(f"Deduplication check failed: {e}")

    job_id = str(uuid.uuid4())
    
    # Check for ffmpeg to decide on format
    has_ffmpeg = shutil.which("ffmpeg") is not None
    
    # Configure yt-dlp with resilience options
    ydl_opts = {
        'format': 'bestvideo[vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/bestvideo[vcodec^=avc1]+bestaudio/best[vcodec^=avc1]/best[ext=mp4][height<=1080]/best[ext=mp4]/best' if has_ffmpeg else 'best[vcodec^=avc1]/best[ext=mp4]/best',
        'outtmpl': str(uploads_dir / f"{job_id}_%(title)s.%(ext)s"),
        'noplaylist': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'convertsubtitles': 'srt',
        'ignoreerrors': 'only_download',  # Don't abort if subtitle download fails (e.g. HTTP 429)
        'socket_timeout': 30,
        'retries': 3,
        'fragment_retries': 3,
        'extractor_retries': 3,
        'file_access_retries': 3,
        'noprogress': True,
        'no_warnings': False,
    }
    
    downloaded_video = None
    downloaded_srt = None
    
    def run_download(opts):
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(payload.url, download=True)
            return ydl.prepare_filename(info)

    try:
        # Attempt 1: With subtitles
        try:
            print(f"[yt-dlp] Downloading with subtitles: {payload.url}")
            downloaded_video = run_download(ydl_opts)
            # Identify SRT file if successful
            base_name = Path(downloaded_video).stem
            potential_srts = list(uploads_dir.glob(f"{base_name}*.srt"))
            if potential_srts:
                downloaded_srt = potential_srts[0]
                print(f"[yt-dlp] Found SRT captions: {downloaded_srt.name}")
        except Exception as e1:
            print(f"[yt-dlp] Download with subtitles failed: {type(e1).__name__}: {e1}")
            print(f"[yt-dlp] Retrying without subtitles...")
            # Attempt 2: Without subtitles
            ydl_opts['writesubtitles'] = False
            ydl_opts['writeautomaticsub'] = False
            ydl_opts.pop('subtitleslangs', None)
            ydl_opts.pop('convertsubtitles', None)
            
            try:
                downloaded_video = run_download(ydl_opts)
            except Exception as e2:
                print(f"[yt-dlp] Second attempt also failed: {type(e2).__name__}: {e2}")
                # Attempt 3: Simplest possible format
                print(f"[yt-dlp] Trying simplest format fallback...")
                ydl_opts['format'] = 'best[ext=mp4]/best'
                downloaded_video = run_download(ydl_opts)
            
    except Exception as e:
        print(f"[yt-dlp] All download attempts failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"YouTube download failed: {e}")
    
    if not downloaded_video or not Path(downloaded_video).exists():
        raise HTTPException(status_code=500, detail="Video download failed or file empty")
        
    saved_path = Path(downloaded_video)
    safe_name = saved_path.name
    
    return process_video_workflow(
        saved_path=saved_path,
        job_id=job_id,
        user_id=payload.user_id,
        query=payload.query,
        original_filename=safe_name,
        provided_srt_path=downloaded_srt,
        input_source=payload.url
    )


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...), user_id: Optional[str] = Form(None), query: Optional[str] = Form(None)):
    """
    Accepts a video upload from the frontend, saves it into backend folder,
    transcribes it to SRT, and writes SRT to a predictable location for
    downstream processing and embeddings.
    """
    # Choose writable directories.
    try:
        base_dir = Path(os.getenv("APP_BASE_DIR") or Path(__file__).resolve().parent)
        uploads_dir = base_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
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

    # Validate file extension (AssemblyAI requires audio/video media)
    allowed_ext = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".webm", ".ogg", ".flac"}
    ext = saved_path.suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Unsupported media type '{ext}'. Please upload audio/video (e.g., .mp3, .wav, .mp4)")

    return process_video_workflow(
        saved_path=saved_path,
        job_id=job_id,
        user_id=user_id,
        query=query,
        original_filename=safe_name
    )

        

class ClipSearchRequest(BaseModel):
    json_path: Optional[str] = None
    job_id: Optional[str] = None
    query: str
    top_k: int = 5
    margin_secs: float = 0.5
    use_windows: bool = True
    window_size: int = 6
    window_stride: int = 2
    expand_neighbors: bool = True
    min_clip_secs: float = 20.0
    max_clip_secs: float = 120.0
    rerank: bool = True

@app.post("/clips/search")
def clips_search(payload: ClipSearchRequest):
    out_dir = REPO_ROOT / "output_data"
    if not payload.json_path and not payload.job_id:
        raise HTTPException(status_code=400, detail="Provide json_path or job_id")
    if payload.json_path:
        json_path = Path(payload.json_path)
    else:
        matches = list(out_dir.glob(f"{payload.job_id}_*.v4.json"))
        if not matches:
            raise HTTPException(status_code=404, detail="JSON not found for job_id")
        json_path = matches[0]
    if not json_path.exists():
        raise HTTPException(status_code=404, detail=f"JSON not found: {json_path}")
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read JSON: {e}")
    sentences = data.get("sentences") or []
    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences in JSON")

    video_path = str(data.get("metadata", {}).get("file"))
    if not video_path:
        raise HTTPException(status_code=400, detail="Video path missing in metadata")
    clips_dir = out_dir / "clips" / (json_path.stem.split("_")[0])
    clips_dir.mkdir(parents=True, exist_ok=True)

    # ── CLEAN QUERY: Strip appended timestamp noise like "[0.0s - 575.1s]" ──
    clean_query = re.sub(r'\s*\[\d+\.?\d*s\s*-\s*\d+\.?\d*s\]\s*$', '', payload.query).strip()
    if clean_query != payload.query:
        print(f"[clips/search] Cleaned query: '{payload.query}' → '{clean_query}'")
    if not clean_query:
        clean_query = payload.query  # fallback if regex ate everything

    # ── Strategy 1: Substantive LLM Search (Groq primary, Gemini fallback) ──
    llm_segments = llm_intelligent_search(clean_query, sentences, top_k=payload.top_k)
    
    if llm_segments:
        print(f"[clips/search] Using LLM reasoning: {len(llm_segments)} segments found")
        results = []
        for rank, seg in enumerate(llm_segments, 1):
            start = max(0.0, seg["start"] - payload.margin_secs)
            end = seg["end"] + payload.margin_secs

            # Clamp to video duration
            try:
                total = float(data.get("metadata", {}).get("duration") or 0)
                if total > 0:
                    end = min(end, total)
            except Exception:
                pass

            # Enforce min/max clip duration
            dur = end - start
            if dur < payload.min_clip_secs:
                end = start + payload.min_clip_secs
                dur = payload.min_clip_secs
            if dur > payload.max_clip_secs:
                end = start + payload.max_clip_secs
                dur = payload.max_clip_secs

            # Re-clamp after adjustment
            try:
                total = float(data.get("metadata", {}).get("duration") or 0)
                if total > 0 and end > total:
                    end = total
                    dur = max(0, end - start)
            except Exception:
                pass

            # Build transcript text for this time range
            text_segment = " ".join([
                s.get("sentence", "") for s in sentences
                if float(s.get("starttime", 0)) >= seg["start"] - 1
                and float(s.get("endtime", 0)) <= seg["end"] + 1
            ])

            # Extract clip with ffmpeg
            clip_path = clips_dir / f"clip_{rank:02d}.mp4"
            import shutil, subprocess
            ff = shutil.which("ffmpeg")
            if not ff:
                try:
                    import imageio_ffmpeg
                    ff = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ff = None
            ok = False
            if ff:
                # FAST: Try stream copy first (instant, no re-encoding)
                cmd_fast = [
                    ff, "-y",
                    "-ss", str(start),
                    "-i", video_path,
                    "-t", str(dur),
                    "-c", "copy",
                    "-movflags", "+faststart",
                    str(clip_path),
                ]
                try:
                    subprocess.run(cmd_fast, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                    ok = clip_path.exists() and clip_path.stat().st_size > 1024
                except Exception:
                    ok = False
                # FALLBACK: Re-encode if stream copy failed
                if not ok:
                    cmd_slow = [
                        ff, "-y",
                        "-ss", str(start),
                        "-i", video_path,
                        "-t", str(dur),
                        "-c:v", "libx264", "-preset", "ultrafast",
                        "-c:a", "aac",
                        "-movflags", "+faststart",
                        str(clip_path),
                    ]
                    try:
                        subprocess.run(cmd_slow, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        ok = clip_path.exists() and clip_path.stat().st_size > 1024
                    except Exception:
                        ok = False
            if not ok:
                clip_path = None
            clip_rel = clip_path.relative_to(out_dir) if clip_path else None

            # Relevance score: map high=1.0, medium=0.7, low=0.4
            relevance_map = {"high": 1.0, "medium": 0.7, "low": 0.4}
            score = relevance_map.get(seg.get("relevance", "medium"), 0.7)

            results.append({
                "rank": rank,
                "score": score,
                "text": text_segment or seg.get("summary", ""),
                "start": start,
                "end": end,
                "clip_path": str(clip_path) if clip_path else None,
                "clip_url": f"/serve-clip?path={clip_rel.as_posix()}" if clip_rel else None,
                "llm_summary": seg.get("summary"),
            })

        # Generate topic explanation
        topic_explanation = _generate_topic_explanation(clean_query, results)
        return {"results": results, "count": len(results), "topic_explanation": topic_explanation}

    # ── Strategy 2: Embedding-based search (fallback) ──
    print(f"[clips/search] LLM search unavailable or returned no results — using embedding search (query='{clean_query}')")
    
    emb_path = json_path.with_name(json_path.stem.replace(".v4", "") + ".embeddings.json")
    vectors = None
    try:
        if emb_path.exists():
            emb = json.loads(emb_path.read_text(encoding="utf-8"))
            vectors = emb.get("vectors")
    except Exception:
        vectors = None
    if vectors is None:
        try:
            vecs = get_model().encode([s["sentence"] for s in sentences])
            vectors = [list(map(float, v)) for v in vecs]
            try:
                emb_out = {
                    "vectors": vectors,
                    "sentences": sentences,
                }
                emb_path.write_text(json.dumps(emb_out, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")
    import math
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))
    def norm(a):
        return math.sqrt(sum(x*x for x in a))
    q_vec = list(map(float, get_model().encode([clean_query])[0]))
    
    # Advanced Search Logic (sliding window)
    candidates = []
    if payload.use_windows and len(vectors) >= payload.window_size:
        W = max(1, int(payload.window_size))
        S = max(1, int(payload.window_stride))
        for i in range(0, len(vectors) - W + 1, S):
            avg = [0.0]*len(vectors[0])
            for j in range(i, i+W):
                v = vectors[j]
                for k in range(len(v)):
                    avg[k] += v[k]
            for k in range(len(avg)):
                avg[k] /= float(W)
            nv = norm(avg) or 1e-9
            nq = norm(q_vec) or 1e-9
            sim = dot(avg, q_vec) / (nv * nq)
            candidates.append((sim, i, i+W-1))
    else:
        for i, v in enumerate(vectors):
            nv = norm(v) or 1e-9
            nq = norm(q_vec) or 1e-9
            sim = dot(v, q_vec) / (nv * nq)
            candidates.append((sim, i, i))

    candidates.sort(reverse=True)
    preselect = candidates[: min(len(candidates), max(payload.top_k*5, 25))]

    # Expand neighbors
    if payload.expand_neighbors and preselect:
        merged = []
        used = set()
        for idx, (sc, a, b) in enumerate(preselect):
            if idx in used:
                continue
            cur_a, cur_b, cur_sc = a, b, sc
            for j in range(idx+1, len(preselect)):
                s2, a2, b2 = preselect[j]
                if a2 <= cur_b + 1:
                    cur_b = max(cur_b, b2)
                    cur_sc = max(cur_sc, s2)
                    used.add(j)
            merged.append((cur_sc, cur_a, cur_b))
        preselect = merged

    # Reranking
    if payload.rerank:
        ce = get_cross_encoder()
        if ce is not None and preselect:
            pairs = []
            for _, a, b in preselect:
                txt = " ".join([sentences[t].get("sentence", "") for t in range(a, b+1)])
                pairs.append((payload.query, txt))
            try:
                ce_scores = ce.predict(pairs)
                preselect = [(float(ce_scores[i]), preselect[i][1], preselect[i][2]) for i in range(len(preselect))]
                preselect.sort(reverse=True)
            except Exception:
                pass

    top = preselect[: max(1, min(payload.top_k, len(preselect)))]

    results = []
    
    for rank, (score, a, b) in enumerate(top, 1):
        s0 = sentences[a]
        s1 = sentences[b]
        try:
            start = max(0.0, float(s0.get("starttime", "0")) - payload.margin_secs)
            end = max(start, float(s1.get("endtime", "0")) + payload.margin_secs)
        except Exception:
            start, end = 0.0, 0.0
        try:
            total = float(data.get("metadata", {}).get("duration") or 0)
            if total > 0:
                end = min(end, total)
        except Exception:
            pass
            
        dur = max(payload.min_clip_secs, end - start)
        if end - start < dur:
            end = start + dur
        if dur > payload.max_clip_secs:
            end = start + payload.max_clip_secs
            dur = payload.max_clip_secs
        
        try:
             total = float(data.get("metadata", {}).get("duration") or 0)
             if total > 0 and end > total:
                 end = total
                 dur = max(0, end - start)
        except Exception:
             pass

        clip_path = clips_dir / f"clip_{rank:02d}.mp4"
        import shutil, subprocess
        ff = shutil.which("ffmpeg")
        if not ff:
            try:
                import imageio_ffmpeg
                ff = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ff = None
        ok = False
        if ff:
            # FAST: Try stream copy first (instant, no re-encoding)
            cmd_fast = [
                ff, "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(dur),
                "-c", "copy",
                "-movflags", "+faststart",
                str(clip_path),
            ]
            try:
                subprocess.run(cmd_fast, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                ok = clip_path.exists() and clip_path.stat().st_size > 1024
            except Exception:
                ok = False
            # FALLBACK: Re-encode if stream copy failed
            if not ok:
                cmd_slow = [
                    ff, "-y",
                    "-ss", str(start),
                    "-i", video_path,
                    "-t", str(dur),
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    str(clip_path),
                ]
                try:
                    subprocess.run(cmd_slow, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    ok = clip_path.exists() and clip_path.stat().st_size > 1024
                except Exception:
                    ok = False
        if not ok:
            clip_path = None
        clip_rel = clip_path.relative_to(out_dir) if clip_path else None
        
        text_segment = " ".join([sentences[t].get("sentence") for t in range(a, b+1)])
        
        results.append({
            "rank": rank,
            "score": score,
            "text": text_segment,
            "start": start,
            "end": end,
            "clip_path": str(clip_path) if clip_path else None,
            "clip_url": f"/serve-clip?path={clip_rel.as_posix()}" if clip_rel else None,
            "llm_summary": None,
        })

    # LLM refinement: enrich embedding-based results with summaries AND filter irrelevant
    try:
        llm_candidates = [
            {"index": i, "text": r["text"], "start": r["start"], "end": r["end"]}
            for i, r in enumerate(results)
        ]
        llm_results = refine_with_llm(clean_query, llm_candidates)
        llm_map = {item["segment_index"]: item for item in llm_results}
        
        filtered_results = []
        for i, r in enumerate(results):
            if i in llm_map:
                llm_item = llm_map[i]
                r["llm_summary"] = llm_item.get("summary")
                # Filter out segments the LLM says are NOT relevant
                if llm_item.get("relevant", True) is False:
                    print(f"[LLM Filter] Removing clip #{r['rank']} — marked as irrelevant")
                    continue
            filtered_results.append(r)
        
        # Re-rank remaining results
        for new_rank, r in enumerate(filtered_results, 1):
            r["rank"] = new_rank
        results = filtered_results
        print(f"[LLM Filter] {len(results)} clips survived relevance filter")
    except Exception as e:
        print(f"LLM enrichment failed in clips_search: {e}")

    # Generate topic explanation
    topic_explanation = _generate_topic_explanation(clean_query, results)

    return {"results": results, "count": len(results), "topic_explanation": topic_explanation}

class UploadAndSearchResponse(BaseModel):
    job_id: str
    json_path: str
    srt_path: str
    results: list
    count: int
    topic_explanation: str = ""

@app.post("/upload-and-search")
async def upload_and_search(
    file: UploadFile = File(...),
    query: str = Form(...),
    top_k: int = Form(5),
    margin_secs: float = Form(0.5),
    user_id: Optional[str] = Form(None),
    use_windows: bool = Form(True),
    window_size: int = Form(5),
    window_stride: int = Form(2),
    expand_neighbors: bool = Form(True),
    min_clip_secs: float = Form(20.0),
    max_clip_secs: float = Form(120.0),
    rerank: bool = Form(True),
):
    try:
        base_dir = Path(os.getenv("APP_BASE_DIR") or Path(__file__).resolve().parent)
        uploads_dir = base_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        output_dir = REPO_ROOT / "output_data"
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    safe_name = Path(file.filename or "uploaded.bin").name
    job_id = str(uuid.uuid4())
    saved_path = uploads_dir / f"{job_id}_{safe_name}"
    try:
        with saved_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    allowed_ext = {".mp3", ".wav", ".m4a", ".mp4", ".mov", ".webm", ".ogg", ".flac"}
    ext = saved_path.suffix.lower()
    if ext not in allowed_ext:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported media type '{ext}'. Please upload audio/video (e.g., .mp3, .wav, .mp4)"
        )
    srt_out = output_dir / f"{saved_path.stem}.srt"
    try:
        result = generate_transcript_from_video(
            source=str(saved_path),
            output_srt_path=str(srt_out),
            language_code=None,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")
    # Resolve SRT text: prefer the in-result srt_text, then read from disk, then plain text
    srt_text = result.get("srt_text") or ""
    if not srt_text and result.get("srt_path"):
        try:
            srt_text = Path(str(result["srt_path"])).read_text(encoding="utf-8", errors="replace")
        except Exception:
            srt_text = ""
    if not srt_text and srt_out.exists():
        try:
            srt_text = srt_out.read_text(encoding="utf-8", errors="replace")
        except Exception:
            srt_text = ""
    if not srt_text:
        # Last resort: use raw transcript text (non-SRT) to build a single sentence block
        srt_text = result.get("text", "")
    sentences = parse_srt_blocks(srt_text)
    # If SRT parsing yielded nothing but we have plain text, synthesize a single sentence entry
    if not sentences and result.get("text", "").strip():
        sentences = [{
            "sentence": result["text"].strip(),
            "starttime": "0.00",
            "endtime": "0.00",
            "verbs": []
        }]

    # --- OCR: Extract text from video frames (runs on ENTIRE video) ---
    ocr_text_data = []
    try:
        from ocr_utils import extract_all_video_frames, run_ocr_on_frames as ocr_run
        frames_dir = output_dir / "frames" / job_id
        print(f"[OCR] Starting full-video OCR extraction...")
        frame_count = extract_all_video_frames(saved_path, frames_dir)
        if frame_count > 0:
            ocr_text_data = ocr_run(frames_dir)
            print(f"[OCR] Completed: {len(ocr_text_data)} frames contained readable text")
        else:
            print("[OCR] No frames could be extracted from video")

        # Merge OCR text into transcript sentences
        if ocr_text_data:
            for ocr_item in ocr_text_data:
                ts = ocr_item["timestamp"]
                text = ocr_item["text"]
                best_match = None
                best_dist = float('inf')
                for s in sentences:
                    s_start = float(s["starttime"])
                    s_end = float(s["endtime"])
                    if s_start <= ts <= s_end:
                        best_match = s
                        break
                    dist = min(abs(s_start - ts), abs(s_end - ts))
                    if dist < 5.0 and dist < best_dist:
                        best_dist = dist
                        best_match = s
                if best_match is not None:
                    best_match["sentence"] += f" [On Screen: {text}]"
                else:
                    sentences.append({
                        "sentence": f"[Visual Content]: {text}",
                        "starttime": f"{ts:.2f}",
                        "endtime": f"{ts+2.0:.2f}",
                        "verbs": ["visual_ocr"]
                    })
            sentences.sort(key=lambda x: float(x["starttime"]))
            print(f"[OCR] Merged {len(ocr_text_data)} OCR results into transcript")
    except Exception as e:
        import traceback
        print(f"[OCR ERROR] OCR in upload-and-search failed: {e}")
        traceback.print_exc()

    duration_seconds = sentences[-1]["endtime"] if sentences else "0.00"
    if not sentences:
        raise HTTPException(status_code=502, detail="Transcription produced no content. The video may be silent or the transcript empty.")
    meta = build_metadata({}, srt_path=str(saved_path.with_suffix('.srt')), video_path=str(saved_path))
    if duration_seconds and meta.get("duration") in (None, "N/A"):
        meta["duration"] = duration_seconds
    data = {"metadata": meta, "sentences": sentences}
    json_path = output_dir / f"{saved_path.stem}.v4.json"
    try:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        emb_vecs = [list(map(float, v)) for v in get_model().encode([s["sentence"] for s in sentences])]
        emb = {"vectors": emb_vecs, "sentences": sentences}
        (output_dir / f"{saved_path.stem}.embeddings.json").write_text(
            json.dumps(emb, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as emb_err:
        print(f"[upload-and-search] Embedding generation failed: {emb_err}")
        import traceback; traceback.print_exc()
    # Persist to supabase
    try:
        if supabase is not None:
            if user_id:
                meta_payload = {
                    "id": job_id,
                    "job_id": job_id,
                    "user_id": user_id,
                    "title": meta.get("title") or safe_name,
                    "original_filename": safe_name,
                    "video_url": str(saved_path),
                    "file_size": int(saved_path.stat().st_size),
                    "duration": float(meta.get("duration") or 0),
                    "metadata": {
                        "json_path": str(json_path),
                        "srt_path": str(srt_out),
                    },
                    "status": "completed",
                }
                try:
                    supabase.table("user_videos").upsert(meta_payload).execute()
                except Exception as e:
                    print("user_videos upsert failed:", e)
                try:
                    vchk = supabase.table("user_videos").select("id").eq("id", job_id).limit(1).execute()
                    if not (vchk.data or []):
                        print("user_videos verify missing row:", job_id)
                except Exception as e:
                    print("user_videos verify failed:", e)
            try:
                import numpy as np
                avg_vec = np.array(emb_vecs, dtype=float).mean(axis=0).tolist() if 'emb_vecs' in locals() and emb_vecs else None
                transcript_text = " ".join([s.get("sentence", "") for s in sentences])
                transcript_vec = None
                try:
                    transcript_vec = list(map(float, get_model().encode([transcript_text])[0])) if transcript_text else None
                except Exception:
                    transcript_vec = None
                ve_row = {
                    "job_id": job_id,
                    "video_url": str(saved_path),
                    "title": meta.get("title") or safe_name,
                    "description": None,
                    "duration": int(float(meta.get("duration") or 0)),
                    "thumbnail_url": None,
                    "transcript": transcript_text or None,
                    "transcript_id": result.get("id", "") or None,
                    "video_embedding": avg_vec,
                    "transcript_embedding": transcript_vec,
                    "audio_url": None,
                    "frames_path": None,
                }
                supabase.table("video_embeddings").insert(ve_row).execute()
            except Exception:
                pass
            try:
                emb_rows = []
                for idx, s in enumerate(sentences):
                    emb_rows.append({
                        "job_id": job_id,
                        "sentence_index": idx,
                        "text": s.get("sentence"),
                        "start": float(s.get("starttime", 0)),
                        "end": float(s.get("endtime", 0)),
                        "embedding": [float(x) for x in emb_vecs[idx]] if 'emb_vecs' in locals() else None,
                    })
                if emb_rows:
                    batch = 200
                    for i in range(0, len(emb_rows), batch):
                        try:
                            supabase.table("video_sentence_embeddings").insert(emb_rows[i:i+batch]).execute()
                        except Exception:
                            pass
            except Exception:
                pass
            if user_id:
                try:
                    # Ensure profile exists
                    try:
                        supabase.table("profiles").upsert({"id": user_id}, on_conflict="id").execute()
                    except Exception:
                        pass
                    supabase.table("processing_history").insert({
                        "user_id": user_id,
                        "video_id": job_id,
                        "module": "summarization",
                        "input_type": "file",
                        "input_url": safe_name,
                        "query": query,
                        "status": "completed",
                    }).execute()
                except Exception:
                    pass
    except Exception:
        pass

    try:
        payload = ClipSearchRequest(
            json_path=str(json_path), 
            job_id=job_id, 
            query=query, 
            top_k=top_k, 
            margin_secs=margin_secs,
            use_windows=use_windows,
            window_size=window_size,
            window_stride=window_stride,
            expand_neighbors=expand_neighbors,
            min_clip_secs=min_clip_secs,
            max_clip_secs=max_clip_secs,
            rerank=rerank
        )
        r = clips_search(payload)
        return UploadAndSearchResponse(job_id=job_id, json_path=str(json_path), srt_path=str(srt_out), results=r["results"], count=r["count"], topic_explanation=r.get("topic_explanation", ""))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline failed at clip search stage: {e}") 

class DbSearchRequest(BaseModel):
    job_id: str
    query: str
    top_k: int = 5
    margin_secs: float = 0.5
    window_size: int = 5
    window_stride: int = 2
    use_windows: bool = True
    rerank: bool = True
    min_clip_secs: float = 20.0
    max_clip_secs: float = 120.0
    expand_neighbors: bool = True

class _LRUCache:
    def __init__(self, capacity: int = 128):
        self.capacity = capacity
        self.store = {}
        self.order = []
    def get(self, key):
        if key in self.store:
            try:
                self.order.remove(key)
            except Exception:
                pass
            self.order.insert(0, key)
            return self.store[key]
        return None
    def put(self, key, value):
        if key in self.store:
            self.store[key] = value
            try:
                self.order.remove(key)
            except Exception:
                pass
            self.order.insert(0, key)
            return
        self.store[key] = value
        self.order.insert(0, key)
        while len(self.order) > self.capacity:
            old = self.order.pop()
            try:
                del self.store[old]
            except Exception:
                pass

_SENTENCE_EMB_CACHE = _LRUCache(128)
_VIDEO_EMB_CACHE = _LRUCache(256)
cross_encoder = None

def get_cross_encoder():
    global cross_encoder
    if cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception:
            cross_encoder = None
    return cross_encoder

# --- LLM Provider: Groq (primary) + Gemini (fallback) ---
# Groq free tier: 30 RPM, 14,400 RPD with Llama 3.3 70B
# Sign up at https://console.groq.com for a free API key

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Best free model on Groq

def _parse_llm_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return json.loads(text)


def _llm_generate(prompt: str, max_tokens: int = 4096) -> str:
    """
    Unified LLM generation function. Tries providers in order:
    1. Groq (GROQ_API_KEY) — free, fast, generous limits
    2. Gemini (GOOGLE_API_KEY) — fallback
    
    Returns the response text, or raises an exception if all providers fail.
    """
    errors = []
    
    # --- Provider 1: Groq ---
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if groq_key:
        try:
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            
            # Retry with backoff for transient rate limits
            for attempt in range(3):
                resp = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                elif resp.status_code == 429:
                    # Rate limited
                    retry_after = int(resp.headers.get("retry-after", 4 * (2 ** attempt)))
                    print(f"[Groq] Rate limited — retrying in {retry_after}s (attempt {attempt+1}/3)")
                    time.sleep(retry_after)
                    continue
                else:
                    err_msg = resp.text[:200]
                    errors.append(f"Groq HTTP {resp.status_code}: {err_msg}")
                    break
            else:
                errors.append("Groq: all retries exhausted")
        except Exception as e:
            errors.append(f"Groq error: {e}")
    
    # --- Provider 2: Gemini (fallback) ---
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if google_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            err_str = str(e)
            if "PerDay" in err_str or "limit: 0" in err_str:
                errors.append("Gemini: daily quota exhausted")
            else:
                errors.append(f"Gemini error: {e}")
    
    # All providers failed
    if not groq_key and not google_key:
        raise RuntimeError("No LLM API key set. Set GROQ_API_KEY (recommended) or GOOGLE_API_KEY.")
    raise RuntimeError(f"All LLM providers failed: {'; '.join(errors)}")


def llm_intelligent_search(query: str, sentences: list, top_k: int = 10) -> list:
    """
    Use LLM to reason about the FULL transcript and find ALL segments
    that match the user's query. Works with Groq (primary) or Gemini (fallback).
    
    Uses strict semantic matching — segments must ACTUALLY discuss the topic,
    not just contain similar-sounding keywords.
    
    Returns:
        List of dicts: [{"start": float, "end": float, "summary": str, "relevance": str}]
        Returns empty list if LLM is unavailable (caller should fall back to embeddings).
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not groq_key and not google_key:
        print("[LLM] No API key set (GROQ_API_KEY or GOOGLE_API_KEY) — falling back to embedding search")
        return []
    
    # Build a condensed transcript with timestamps
    CHUNK_SIZE = 5
    chunks = []
    for i in range(0, len(sentences), CHUNK_SIZE):
        group = sentences[i:i+CHUNK_SIZE]
        start_t = group[0].get("starttime", "0")
        end_t = group[-1].get("endtime", "0")
        text = " ".join(s.get("sentence", "") for s in group)
        chunks.append(f"[{start_t}s - {end_t}s]: {text}")
    
    transcript_text = "\n".join(chunks)
    
    # Limit to ~12000 chars for token safety
    if len(transcript_text) > 12000:
        transcript_text = transcript_text[:12000] + "\n[... transcript truncated ...]"
    
    prompt = f"""You are an expert video content analyst performing STRICT semantic search on a video transcript.

Your task: Find segments where the video ACTUALLY DISCUSSES the user's query topic in a meaningful, substantive way.

CRITICAL RULES FOR ACCURACY:
1. A segment is RELEVANT only if the speaker is genuinely explaining, discussing, or demonstrating the queried concept.
2. A segment is NOT RELEVANT if it merely contains a word that looks or sounds similar to the query but discusses something entirely different.
   - Example: If query is "vector database", a segment mentioning "vectorless" or "vector graphics" is NOT relevant unless it's actually about vector databases.
   - Example: If query is "RAG (Retrieval Augmented Generation)", only return segments where RAG is actually explained or discussed, not where "rag" appears as a common word.
3. Consider the FULL CONTEXT of each segment — does the surrounding conversation support that this is about the queried topic?
4. Prefer segments with EXPLANATIONS, DEFINITIONS, or DEMONSTRATIONS of the concept over mere mentions.
5. If the video does NOT substantively discuss the queried topic at all, return an EMPTY segments array — do NOT force irrelevant matches.
6. Merge nearby segments (within 10 seconds) into a single result.
7. Return up to {top_k} segments, ranked by relevance (most relevant first).
8. Each segment should have precise start/end timestamps from the transcript.
9. Include [On Screen] and [Visual Content] text in your analysis if present.

User Query: "{query}"

Full Video Transcript (with timestamps):
{transcript_text}

Before responding, think step-by-step:
- What EXACTLY is the user looking for?
- Which segments genuinely discuss this specific topic (not just contain similar words)?
- Would a human watching these segments feel their question was answered?

Respond ONLY with valid JSON:
{{
  "reasoning": "Explain what the user is searching for and your evaluation of which segments truly match vs. which are false positives",
  "segments": [
    {{
      "start": 12.5,
      "end": 45.3,
      "summary": "2-3 sentence description of what is ACTUALLY discussed in this segment and how it relates to the query",
      "relevance": "high/medium/low"
    }}
  ]
}}"""
    
    try:
        provider = "Groq" if groq_key else "Gemini"
        print(f"[{provider}] Searching transcript ({len(sentences)} sentences) for: {query[:80]}...")
        
        response_text = _llm_generate(prompt)
        parsed = _parse_llm_json(response_text)
        segments = parsed.get("segments", [])
        reasoning = parsed.get("reasoning", "")
        if reasoning:
            print(f"[{provider}] Reasoning: {reasoning[:200]}")
        print(f"[{provider}] Found {len(segments)} relevant segments")
        
        # Validate and clean segments
        valid_segments = []
        for seg in segments:
            try:
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                if end > start:
                    valid_segments.append({
                        "start": start,
                        "end": end,
                        "summary": seg.get("summary", ""),
                        "relevance": seg.get("relevance", "medium"),
                    })
            except (ValueError, TypeError):
                continue
        
        return valid_segments[:top_k]
    except Exception as e:
        print(f"[LLM] Intelligent search failed: {e}")
        return []


def refine_with_llm(query: str, candidates: list) -> list:
    """
    Send candidate transcript segments to LLM for relevance assessment and summary.
    Each candidate: {"index": int, "text": str, "start": float, "end": float}
    Returns list of {"segment_index": int, "start": float, "end": float, "summary": str, "relevant": bool}
    Falls back to empty list on any failure.
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not groq_key and not google_key:
        return []
    
    segments_text = ""
    for c in candidates:
        segments_text += f"[Segment {c['index']}] {c['start']:.1f}s - {c['end']:.1f}s: \"{c['text']}\"\n"
    
    prompt = f"""You are a strict video content relevance judge. Given the user's query and transcript segments, you must:
1. Determine if each segment is TRULY RELEVANT to the query (not just containing similar keywords)
2. Provide a concise summary for relevant segments

CRITICAL: A segment is RELEVANT only if it genuinely discusses, explains, or demonstrates the queried concept.
A segment is NOT RELEVANT if it just contains words that look similar but the actual discussion is about something else.
For example, if the query is about "vector databases" and a segment talks about "vectorless RAG" (a different concept), mark it as NOT relevant.

User Query: "{query}"

Transcript Segments:
{segments_text}

Respond ONLY with valid JSON:
{{
  "results": [
    {{
      "segment_index": 0,
      "start": 12.5,
      "end": 45.3,
      "summary": "Brief description of what this segment discusses and how it relates to the query",
      "relevant": true
    }}
  ]
}}"""
    
    try:
        response_text = _llm_generate(prompt, max_tokens=2048)
        parsed = _parse_llm_json(response_text)
        return parsed.get("results", [])
    except Exception as e:
        print(f"LLM refinement failed: {e}")
        return []


def _generate_topic_explanation(query: str, results: list) -> str:
    """
    Generate a comprehensive summary explanation about the user's query topic,
    based on the found video segments. Shown below the video clips as a
    "Topic Overview" section.
    """
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not groq_key and not google_key:
        return ""
    if not results:
        return ""

    segments_context = ""
    for r in results[:5]:
        text = r.get("text", "") or r.get("llm_summary", "")
        if text:
            segments_context += f"- [{r['start']:.1f}s - {r['end']:.1f}s]: {text[:400]}\n"
    if not segments_context:
        return ""

    prompt = f"""You are an expert content summarizer. Based on the video segments found for the user's query, write a comprehensive and informative summary.

User Query: "{query}"

Found Video Segments:
{segments_context}

Write a clear, well-structured response that:
1. Directly addresses the user's query based on what the video actually discusses
2. Summarizes the key points, definitions, and explanations found in the video
3. Highlights the most important takeaways
4. Notes any practical examples or demonstrations if present
5. Mentions which parts of the video are most valuable for the user's query

IMPORTANT: Write in a natural, informative tone. Use 4-6 sentences. If the video segments don't strongly relate to the query, say so honestly rather than fabricating relevance.

Respond ONLY with valid JSON:
{{
  "explanation": "Your comprehensive summary here..."
}}"""

    try:
        response_text = _llm_generate(prompt, max_tokens=1024)
        parsed = _parse_llm_json(response_text)
        return parsed.get("explanation", "")
    except Exception as e:
        print(f"[LLM] Topic explanation generation failed: {e}")
        return ""


@app.post("/clips/search-db")
def clips_search_db(payload: DbSearchRequest):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        # Locate JSON by job_id instead of user_videos
        out_dir = REPO_ROOT / "output_data"
        matches = list(out_dir.glob(f"{payload.job_id}_*.v4.json"))
        if not matches:
            raise HTTPException(status_code=404, detail="JSON not found for job_id")
        json_path = matches[0]
        data = json.loads(json_path.read_text(encoding="utf-8"))
        sentences = data.get("sentences") or []
        vecs = _SENTENCE_EMB_CACHE.get(payload.job_id)
        if vecs is None:
            e = supabase.table("video_sentence_embeddings").select("sentence_index, embedding").eq("job_id", payload.job_id).order("sentence_index").execute()
            rows = e.data or []
            vecs = []
            for r in rows:
                val = r.get("embedding") or []
                if isinstance(val, str):
                    try:
                        import json as _json
                        val = _json.loads(val)
                    except Exception:
                        val = []
                try:
                    vecs.append([float(x) for x in (val or [])])
                except Exception:
                    vecs.append([])
            if vecs:
                _SENTENCE_EMB_CACHE.put(payload.job_id, vecs)
        if not vecs:
            # Fallback: compute and optionally store
            try:
                vecs = [list(map(float, v)) for v in get_model().encode([s["sentence"] for s in sentences])]
            except Exception as ex:
                raise HTTPException(status_code=500, detail=f"Embedding failed: {ex}")
        import math
        def dot(a, b):
            return sum(x*y for x, y in zip(a, b))
        def norm(a):
            return math.sqrt(sum(x*x for x in a))
        q_vec = list(map(float, get_model().encode([payload.query])[0]))
        candidates = []
        if payload.use_windows and len(vecs) >= payload.window_size:
            W = max(1, int(payload.window_size))
            S = max(1, int(payload.window_stride))
            for i in range(0, len(vecs) - W + 1, S):
                avg = [0.0]*len(vecs[0])
                for j in range(i, i+W):
                    v = vecs[j]
                    for k in range(len(v)):
                        avg[k] += v[k]
                for k in range(len(avg)):
                    avg[k] /= float(W)
                nv = norm(avg) or 1e-9
                nq = norm(q_vec) or 1e-9
                sim = dot(avg, q_vec) / (nv * nq)
                candidates.append((sim, i, i+W-1))
        else:
            for i, v in enumerate(vecs):
                nv = norm(v) or 1e-9
                nq = norm(q_vec) or 1e-9
                sim = dot(v, q_vec) / (nv * nq)
                candidates.append((sim, i, i))
        candidates.sort(reverse=True)
        preselect = candidates[: min(len(candidates), max(payload.top_k*5, 25))]
        # Merge adjacent windows if requested
        if payload.expand_neighbors and preselect:
            merged = []
            used = set()
            for idx, (sc, a, b) in enumerate(preselect):
                if idx in used:
                    continue
                cur_a, cur_b, cur_sc = a, b, sc
                # try to grow by adding nearby windows
                for j in range(idx+1, len(preselect)):
                    s2, a2, b2 = preselect[j]
                    # contiguous or overlapping
                    if a2 <= cur_b + 1:
                        cur_b = max(cur_b, b2)
                        cur_sc = max(cur_sc, s2)
                        used.add(j)
                merged.append((cur_sc, cur_a, cur_b))
            preselect = merged
        if payload.rerank:
            ce = get_cross_encoder()
            if ce is not None and preselect:
                pairs = []
                for _, a, b in preselect:
                    txt = " ".join([sentences[t].get("sentence", "") for t in range(a, b+1)])
                    pairs.append((payload.query, txt))
                try:
                    ce_scores = ce.predict(pairs)
                    preselect = [(float(ce_scores[i]), preselect[i][1], preselect[i][2]) for i in range(len(preselect))]
                    preselect.sort(reverse=True)
                except Exception:
                    pass
        top = preselect[: max(1, min(payload.top_k, len(preselect)))]
        # Resolve source video path via video_embeddings
        ve = supabase.table("video_embeddings").select("video_url").eq("job_id", payload.job_id).limit(1).execute()
        vrows = ve.data or []
        video_path = (vrows[0] or {}).get("video_url") if vrows else None
        if not video_path:
            # Fallback to metadata file path
            video_path = str(data.get("metadata", {}).get("file"))
        if not video_path:
            raise HTTPException(status_code=404, detail="Video path not found")
        clips_dir = out_dir / "clips" / payload.job_id
        clips_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for rank, (score, a, b) in enumerate(top, 1):
            s0 = sentences[a]
            s1 = sentences[b]
            try:
                start = max(0.0, float(s0.get("starttime", "0")) - payload.margin_secs)
                end = max(start, float(s1.get("endtime", "0")) + payload.margin_secs)
            except Exception:
                start, end = 0.0, 0.0
            try:
                total = float(data.get("metadata", {}).get("duration") or 0)
                if total > 0:
                    end = min(end, total)
            except Exception:
                pass
            # enforce min/max duration
            dur = max(payload.min_clip_secs, end - start)
            if dur > payload.max_clip_secs:
                end = start + payload.max_clip_secs
                dur = payload.max_clip_secs
            clip_path = clips_dir / f"clip_{rank:02d}.mp4"
            import shutil, subprocess
            ff = shutil.which("ffmpeg")
            if not ff:
                try:
                    import imageio_ffmpeg
                    ff = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ff = None
            ok = False
            if ff:
                cmd = [ff, "-y", "-ss", str(start), "-i", str(video_path), "-t", str(dur), "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", str(clip_path)]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    ok = clip_path.exists() and clip_path.stat().st_size > 1024
                except Exception:
                    ok = False
            if not ok:
                clip_path = None
            clip_rel = clip_path.relative_to(out_dir) if clip_path else None
            results.append({
                "rank": rank,
                "score": score,
                "text": " ".join([sentences[t].get("sentence") for t in range(a, b+1)]),
                "start": start,
                "end": start + dur,
                "clip_path": str(clip_path) if clip_path else None,
                "clip_url": f"/static/{clip_rel.as_posix()}" if clip_rel else None,
                "llm_summary": None,
            })

        # LLM refinement: enrich results with AI-generated summaries
        try:
            llm_candidates = [
                {"index": i, "text": r["text"], "start": r["start"], "end": r["end"]}
                for i, r in enumerate(results)
            ]
            llm_results = refine_with_llm(payload.query, llm_candidates)
            llm_map = {item["segment_index"]: item for item in llm_results}
            for i, r in enumerate(results):
                if i in llm_map:
                    r["llm_summary"] = llm_map[i].get("summary")
        except Exception as e:
            print(f"LLM enrichment failed in clips_search_db: {e}")

        return {"results": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(user_id: str, page: int = 1, page_size: int = 20, q: Optional[str] = None, include_embeddings: bool = False):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        p = max(1, int(page))
        ps = max(1, min(int(page_size), 100))
        start = (p - 1) * ps
        end = start + ps - 1
        sel = supabase.table("processing_history").select("id,created_at,query,status,video_id,module").eq("user_id", user_id)
        if q and q.strip():
            try:
                sel = sel.ilike("query", f"%{q}%")
            except Exception:
                pass
        # Use proper order signature for supabase-py (desc=True for latest first)
        sel = sel.order("created_at", desc=True)
        # Apply pagination window
        try:
            hist = sel.range(start, end).execute()
        except Exception:
            # Fallback when range is unsupported: use limit
            hist = sel.limit(ps).execute()
        hrows = hist.data or []
        vids = [r.get("video_id") for r in hrows if r.get("video_id")]
        vmeta = {}
        vemb = {}
        if vids:
            vm = supabase.table("video_embeddings").select("job_id,title,video_url,duration,transcript_embedding,video_embedding").in_("job_id", vids).execute()
            for r in (vm.data or []):
                vmeta[r.get("job_id")] = r
            if include_embeddings:
                for key, val in vmeta.items():
                    vemb[key] = {"video_embedding": val.get("video_embedding"), "transcript_embedding": val.get("transcript_embedding")}
                    _VIDEO_EMB_CACHE.put(key, vemb[key])
        items = []
        for r in hrows:
            vid = r.get("video_id")
            items.append({
                "id": r.get("id"),
                "created_at": r.get("created_at"),
                "query": r.get("query"),
                "status": r.get("status"),
                "module": r.get("module"),
                "video": vmeta.get(vid),
                "embeddings": vemb.get(vid) if include_embeddings else None,
            })
        return {"items": items, "page": p, "page_size": ps, "count": len(items)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/download")
async def download_file(path: str):
    """
    Serve files from valid directories (uploads, output_data).
    Path should be relative to backend root or one of the allowed dirs.
    """
    base_dir = Path(os.getenv("APP_BASE_DIR") or Path(__file__).resolve().parent)
    
    # Sanitize path to prevent traversal
    # We expect path to be like "uploads/file.mp4" or "output_data/clips/jobid/clip.mp4"
    safe_path = Path(path).name # This is too aggressive if we have subdirs.
    
    # Better: resolve and check parents
    try:
        # Check if it's an output_data path (which lives in a different root than base_dir)
        out_root = Path(__file__).resolve().parents[2] / "backend"
        if str(path).startswith("output_data") or str(path).startswith("output_data\\"):
             target_path = (out_root / path).resolve()
        else:
             target_path = (base_dir / path).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    allowed_parents = [
        (base_dir / "uploads").resolve(),
        (Path(__file__).resolve().parents[2] / "backend" / "output_data").resolve() 
    ]
    
    is_allowed = False
    for p in allowed_parents:
        # Check if target_path starts with allowed parent path
        if str(target_path).startswith(str(p)):
            is_allowed = True
            break
            
    if not is_allowed:
        # Fallback: check if path is just a filename in uploads (legacy support)
        if (base_dir / "uploads" / path).exists():
             target_path = (base_dir / "uploads" / path).resolve()
             is_allowed = True
        
    if not is_allowed:
         # Debug logging
         print(f"Access denied for path: {target_path}")
         print(f"Allowed parents: {allowed_parents}")
         raise HTTPException(status_code=403, detail="Access denied")
             
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        path=target_path, 
        filename=target_path.name, 
        media_type='application/octet-stream'
    )

@app.post("/compress-video")
async def compress_video(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Compresses an uploaded video using FFmpeg with H.265, scaling to 720p, and CRF 28.
    Returns the compressed video URL and size statistics.
    
    Fixes applied:
    - FFmpeg flag ordering bug: all quality flags now appear BEFORE output path
    - Removed CRF + bitrate conflict on CPU mode (CRF-only now)
    - Scale filter capped at 720p height instead of 1280px width
    - Lowered GPU bitrate targets to prevent large file bloat
    - Switched CPU preset from 'fast' to 'medium' for better compression ratio
    - Reduced audio bitrate from 128k to 96k
    - Explicit x265 thread pool limit to avoid Kaggle threading issues
    """
    import subprocess

    # -------------------------------------------------------------------------
    # 1. Directory Setup
    # -------------------------------------------------------------------------
    try:
        uploads_dir = REPO_ROOT / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        output_dir = REPO_ROOT / "output_data" / "compressed"
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Directory setup failed: {e}")

    # -------------------------------------------------------------------------
    # 2. File Save
    # -------------------------------------------------------------------------
    safe_name = Path(file.filename or "video.mp4").name
    job_id    = str(uuid.uuid4())
    input_path  = uploads_dir / f"{job_id}_in_{safe_name}"
    output_path = output_dir  / f"{job_id}_compressed_{safe_name}"

    try:
        with input_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    original_size = input_path.stat().st_size
    
    # --- Duration Extraction (ffprobe) ---
    probe_duration = 0.0
    try:
        # Check for ffprobe
        fp = shutil.which("ffprobe")
        if not fp:
            try:
                import imageio_ffmpeg
                # imageio_ffmpeg might provide ffprobe too if we find the binary in the same dir
                fp_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
                fp = str(fp_path.parent / "ffprobe")
                if os.name == 'nt': fp += ".exe"
                if not Path(fp).exists(): fp = None
            except:
                fp = None

        if fp:
            # -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1
            probe_cmd = [
                fp, "-v", "error", 
                "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", 
                str(input_path)
            ]
            p_res = subprocess.run(probe_cmd, capture_output=True, text=True)
            if p_res.returncode == 0:
                probe_duration = float(p_res.stdout.strip())
    except Exception as e:
        print(f"ffprobe duration extraction failed: {e}")

    # -------------------------------------------------------------------------
    # 3. Locate FFmpeg
    # -------------------------------------------------------------------------
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            raise HTTPException(status_code=500, detail="FFmpeg not found on server")

    # -------------------------------------------------------------------------
    # 4. GPU Detection
    # -------------------------------------------------------------------------
    has_gpu   = False
    v_encoder = "libx265"

    try:
        gpu_check = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if gpu_check.returncode == 0:
            codec_check = subprocess.run(
                [ff, "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if b"hevc_nvenc" in codec_check.stdout:
                has_gpu   = True
                v_encoder = "hevc_nvenc"
                print("--- [GPU] NVIDIA HEVC NVENC detected and enabled. ---")
    except Exception as e:
        print(f"--- [GPU] Detection failed, falling back to CPU: {e} ---")

    if not has_gpu:
        print("--- [CPU] Using libx265 (CPU encoding). ---")

    # -------------------------------------------------------------------------
    # 4b. Probe Input Video (bitrate, resolution, codec, audio bitrate)
    #
    #     Adaptive compression: We MUST know the input characteristics
    #     before choosing encoding settings. Without this, already-compressed
    #     or low-bitrate videos get RE-ENCODED at higher quality than the
    #     original, causing file size inflation.
    # -------------------------------------------------------------------------
    input_bitrate_kbps  = 0
    input_height        = 0
    input_width         = 0
    input_codec         = ""
    input_audio_bitrate = 0
    input_audio_codec   = ""

    try:
        fp = shutil.which("ffprobe")
        if not fp:
            try:
                import imageio_ffmpeg
                fp_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
                fp = str(fp_path.parent / ("ffprobe.exe" if os.name == 'nt' else "ffprobe"))
                if not Path(fp).exists():
                    fp = None
            except Exception:
                fp = None

        if fp:
            # Get video stream info: bitrate, height, width, codec
            probe_v = subprocess.run(
                [fp, "-v", "error",
                 "-select_streams", "v:0",
                 "-show_entries", "stream=bit_rate,height,width,codec_name",
                 "-of", "json",
                 str(input_path)],
                capture_output=True, text=True
            )
            if probe_v.returncode == 0:
                import json as _json
                pv = _json.loads(probe_v.stdout)
                streams = pv.get("streams", [])
                if streams:
                    s = streams[0]
                    input_codec  = (s.get("codec_name") or "").lower()
                    input_height = int(s.get("height") or 0)
                    input_width  = int(s.get("width") or 0)
                    vbr = s.get("bit_rate")
                    if vbr and str(vbr).isdigit():
                        input_bitrate_kbps = int(vbr) // 1000

            # If per-stream bitrate not available, compute from file size & duration
            if input_bitrate_kbps == 0 and probe_duration > 0:
                input_bitrate_kbps = int((original_size * 8) / probe_duration / 1000)

            # Get audio stream info
            probe_a = subprocess.run(
                [fp, "-v", "error",
                 "-select_streams", "a:0",
                 "-show_entries", "stream=bit_rate,codec_name",
                 "-of", "json",
                 str(input_path)],
                capture_output=True, text=True
            )
            if probe_a.returncode == 0:
                pa = _json.loads(probe_a.stdout)
                astreams = pa.get("streams", [])
                if astreams:
                    input_audio_codec = (astreams[0].get("codec_name") or "").lower()
                    abr = astreams[0].get("bit_rate")
                    if abr and str(abr).isdigit():
                        input_audio_bitrate = int(abr) // 1000

        print(f"--- [PROBE] codec={input_codec}, {input_width}x{input_height}, "
              f"v_bitrate={input_bitrate_kbps}kbps, "
              f"a_codec={input_audio_codec}, a_bitrate={input_audio_bitrate}kbps ---")
    except Exception as e:
        print(f"--- [PROBE] Input analysis failed (will use defaults): {e} ---")

    # -------------------------------------------------------------------------
    # 5. Build Adaptive FFmpeg Command
    #
    #    KEY INSIGHT: The old code used fixed CRF 28 / fixed bitrate caps
    #    regardless of input. This works great for high-bitrate source
    #    material (camera raws, screen-capture, etc.) but INFLATES files
    #    that are already efficiently encoded at low bitrate.
    #
    #    New approach:
    #    a) Compute a TARGET bitrate that is at most 70% of input bitrate
    #    b) Use CRF as quality floor, but add -maxrate cap to prevent bloat
    #    c) Skip scaling if already <= 720p
    #    d) Copy audio if already AAC and <= 128kbps (avoid inflation)
    #    e) If first pass STILL inflates → retry more aggressively
    #    f) If retry STILL inflates → serve original file
    # -------------------------------------------------------------------------

    # --- Determine if downscaling is needed ---
    needs_scale = input_height > 720 or (input_height == 0)  # unknown → apply scale safely
    vf = "scale=-2:min(720\\,ih)" if needs_scale else None

    # --- Determine target video bitrate (adaptive) ---
    # Goal: produce output at most 70% of input bitrate
    # Floor: never go below 300 kbps (unwatchable below that)
    # Ceiling: never exceed 2500 kbps (diminishing returns above that for 720p)
    if input_bitrate_kbps > 0:
        target_bitrate_kbps = max(300, min(int(input_bitrate_kbps * 0.65), 2500))
        maxrate_kbps        = max(400, min(int(input_bitrate_kbps * 0.80), 3000))
    else:
        # Unknown input → conservative defaults
        target_bitrate_kbps = 1500
        maxrate_kbps        = 2500

    # --- Determine audio handling ---
    # If input audio is already AAC at ≤ 128kbps, just copy it (saves space + time)
    copy_audio = (input_audio_codec == "aac" and 0 < input_audio_bitrate <= 128)
    audio_bitrate = "64k" if input_audio_bitrate > 0 and input_audio_bitrate < 96 else "96k"

    # --- Determine CRF based on input quality ---
    # Higher CRF = smaller file but lower quality
    # For already-compressed content, use CRF 30-32 (since it's already lossy)
    # For high-bitrate content, use CRF 26-28 (more room to compress)
    already_efficient = (
        input_codec in ("hevc", "h265", "av1")
        or (input_codec in ("h264", "avc") and input_bitrate_kbps > 0 and input_bitrate_kbps < 1500)
    )
    if already_efficient:
        crf_value = "30"
        print(f"--- [ADAPT] Input already efficient ({input_codec} @ {input_bitrate_kbps}kbps) → CRF {crf_value} ---")
    elif input_bitrate_kbps > 5000:
        crf_value = "26"
        print(f"--- [ADAPT] High-bitrate input ({input_bitrate_kbps}kbps) → CRF {crf_value} ---")
    else:
        crf_value = "28"
        print(f"--- [ADAPT] Standard input → CRF {crf_value} ---")

    def _build_compress_cmd(out_path, crf, tgt_br, max_br, aggressive=False):
        """Build the FFmpeg command with the given parameters."""
        c = [ff, "-y", "-i", str(input_path)]

        # Video filter (scale) — only if needed
        if vf:
            c += ["-vf", vf]

        if has_gpu:
            c += [
                "-c:v",     "hevc_nvenc",
                "-preset",  "p5" if aggressive else "p4",
                "-rc",      "vbr",
                "-cq",      str(int(crf) + (2 if aggressive else 0)),
                "-b:v",     f"{tgt_br}k",
                "-maxrate", f"{max_br}k",
                "-bufsize", f"{max_br * 2}k",
                "-qmin",    str(int(crf) - 2),
                "-qmax",    str(int(crf) + 8),
            ]
        else:
            # CPU: CRF + maxrate (maxrate prevents bloating above target)
            x265_params = f"pools=4:frame-threads=2:vbv-maxrate={max_br}:vbv-bufsize={max_br * 2}"
            c += [
                "-c:v",         "libx265",
                "-preset",      "slower" if aggressive else "medium",
                "-crf",         crf,
                "-x265-params", x265_params,
            ]

        # Audio
        if copy_audio and not aggressive:
            c += ["-c:a", "copy"]
        else:
            c += ["-c:a", "aac", "-b:a", "64k" if aggressive else audio_bitrate]

        c += ["-movflags", "+faststart", str(out_path)]
        return c

    # -------------------------------------------------------------------------
    # 6. Run Compression (with adaptive retry)
    # -------------------------------------------------------------------------
    mode_label = "HEVC_NVENC" if has_gpu else "LIBX265"
    start_time = time.time()
    attempt = 0
    used_original = False

    for attempt in range(2):  # attempt 0 = normal, attempt 1 = aggressive
        aggressive = (attempt == 1)
        if aggressive:
            print("--- [RETRY] First pass inflated file → retrying with aggressive settings ---")
            # More aggressive: higher CRF, lower bitrate
            retry_crf = str(int(crf_value) + 4)  # e.g. 28 → 32
            retry_tgt = max(200, target_bitrate_kbps // 2)
            retry_max = max(300, maxrate_kbps // 2)
            cmd = _build_compress_cmd(output_path, retry_crf, retry_tgt, retry_max, aggressive=True)
        else:
            cmd = _build_compress_cmd(output_path, crf_value, target_bitrate_kbps, maxrate_kbps, aggressive=False)

        print(f"--- [COMPRESS] Attempt {attempt+1}: {' '.join(cmd)} ---")

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            err_out = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
            if attempt == 0:
                # First attempt failed — try falling back to simpler settings
                print(f"--- [COMPRESS] Attempt 1 failed: {err_out[:200]} — trying fallback ---")
                # Simple fallback: use libx264 (universally supported) with high CRF
                fallback_cmd = [
                    ff, "-y", "-i", str(input_path),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "30",
                    "-c:a", "aac", "-b:a", "96k",
                    "-movflags", "+faststart",
                    str(output_path),
                ]
                try:
                    subprocess.run(fallback_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    mode_label = "LIBX264_FALLBACK"
                except subprocess.CalledProcessError as e2:
                    err2 = e2.stderr.decode("utf-8", errors="replace") if e2.stderr else str(e2)
                    raise HTTPException(status_code=500, detail=f"Compression failed: {err2}")
                break
            else:
                raise HTTPException(status_code=500, detail=f"Compression failed: {err_out}")

        # Check if output exists and is not empty
        if not output_path.exists() or output_path.stat().st_size == 0:
            if attempt == 0:
                continue  # retry
            raise HTTPException(status_code=500, detail="Compression resulted in empty file")

        compressed_size = output_path.stat().st_size

        # Check if compression actually reduced the file
        if compressed_size < original_size:
            print(f"--- [COMPRESS] Attempt {attempt+1} successful: "
                  f"{original_size} → {compressed_size} bytes "
                  f"({100 - compressed_size/original_size*100:.1f}% reduction) ---")
            break
        else:
            print(f"--- [COMPRESS] Attempt {attempt+1}: output ({compressed_size}) >= input ({original_size}) ---")
            if attempt == 0:
                # Delete inflated output, will retry
                try:
                    output_path.unlink()
                except Exception:
                    pass
                continue
            else:
                # Even aggressive pass couldn't help — serve the original
                print("--- [COMPRESS] Both passes inflated file → serving original ---")
                try:
                    shutil.copy2(str(input_path), str(output_path))
                except Exception:
                    pass
                used_original = True
                compressed_size = original_size
                break

    duration = time.time() - start_time

    # -------------------------------------------------------------------------
    # 7. Validate Output & Calculate Stats
    # -------------------------------------------------------------------------
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="Compression resulted in empty file")

    compressed_size = output_path.stat().st_size
    reduction = (
        100 - (compressed_size / original_size * 100)
        if original_size > 0 else 0
    )

    if used_original:
        mode_label += "_PASSTHROUGH"
        print(f"--- [RESULT] Served original file (compression would inflate). Reduction: 0% ---")
    else:
        print(f"--- [RESULT] Final: {original_size} → {compressed_size} bytes, "
              f"reduction: {reduction:.1f}% ---")

    # -------------------------------------------------------------------------
    # 8. Save to Processing History & Video Metadata
    # -------------------------------------------------------------------------
    if supabase is not None:
        try:
            rel_path = output_path.relative_to(REPO_ROOT / "output_data")
            static_url = f"/static/{rel_path.as_posix()}"

            # A. user_videos first (processing_history.video_id FK)
            if user_id:
                try:
                    supabase.table("profiles").upsert(
                        {
                            "id": user_id,
                            "email": f"user_{user_id[:8]}@neuroclip.local",
                            "full_name": "NeuroClip User",
                        },
                        on_conflict="id",
                    ).execute()
                except Exception:
                    pass
                supabase.table("user_videos").upsert({
                    "id":                job_id,
                    "user_id":           user_id,
                    "job_id":            job_id,
                    "title":             safe_name,
                    "original_filename": safe_name,
                    "video_url":         static_url,
                    "file_size":         int(compressed_size),
                    "duration":          float(probe_duration),
                    "status":            "completed",
                    "metadata": {
                        "module": "compression",
                        "encoder": mode_label,
                        "original_size": original_size,
                        "reduction": reduction,
                        "proc_time": duration,
                        "used_original": used_original,
                        "input_codec": input_codec,
                        "input_bitrate_kbps": input_bitrate_kbps,
                    }
                }).execute()
                supabase.table("processing_history").insert({
                    "user_id": user_id,
                    "video_id": job_id,
                    "module": "compression",
                    "input_type": "file",
                    "input_url": safe_name,
                    "query": (
                        f"H.265 / {mode_label} / 720p / "
                        f"Total Proc: {duration:.2f}s / "
                        f"Reduction: {reduction:.1f}%"
                    ),
                    "result_url": static_url,
                    "status": "completed",
                }).execute()
            
        except Exception as e:
            print("Supabase persistence failed for compression job:", e)

    # -------------------------------------------------------------------------
    # 9. Return Response
    # -------------------------------------------------------------------------
    rel_path = output_path.relative_to(REPO_ROOT / "output_data")

    return {
        "job_id":           job_id,
        "original_size":    original_size,
        "compressed_size":  compressed_size,
        "reduction":        round(reduction, 2),
        "duration_seconds": round(duration, 2),
        "encoder":          mode_label,
        "url":              f"/static/{rel_path.as_posix()}",
        "used_original":    used_original,
    }


@app.post("/anonymize-video")
async def anonymize_video_endpoint(
    file: UploadFile = File(...),
    reference_images: List[UploadFile] = File(...),
    user_id: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    match_threshold: float = Form(0.78),
    throttle: int = Form(3),
    grace: int = Form(30),
    min_match_streak: int = Form(2),
    start_sec: Optional[float] = Form(None),
    end_sec: Optional[float] = Form(None),
):
    """
    Anonymize a video by blurring faces that match person(s) in reference images.
    Uses YOLOv8 + optional InsightFace on GPU (Kaggle). Query text is stored for history only.
    """
    if not reference_images:
        raise HTTPException(status_code=400, detail="At least one reference image is required")

    try:
        from blur_service import anonymize_video
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Blur service unavailable. Install blur dependencies: {e}",
        )

    uploads_dir = REPO_ROOT / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    output_dir = REPO_ROOT / "output_data" / "anonymized"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename or "video.mp4").name
    job_id = str(uuid.uuid4())
    input_path = uploads_dir / f"{job_id}_in_{safe_name}"
    refs_dir = uploads_dir / f"{job_id}_refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{job_id}_anonymized_{safe_name}"

    try:
        with input_path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    saved_refs = 0
    for idx, ref in enumerate(reference_images):
        ref_name = Path(ref.filename or f"ref_{idx}.jpg").name
        ref_path = refs_dir / ref_name
        try:
            with ref_path.open("wb") as out:
                shutil.copyfileobj(ref.file, out)
            saved_refs += 1
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reference image save failed: {e}")

    if saved_refs == 0:
        raise HTTPException(status_code=400, detail="No reference images could be saved")

    query_note = (query or "").strip()
    from blur_jobs import get_job, start_blur_job, update_job

    def _run_job() -> None:
        from blur_service import anonymize_video

        def _progress(pct: float, msg: str) -> None:
            update_job(job_id, progress=round(pct, 1), message=msg)

        try:
            update_job(job_id, message="Building face signatures", progress=2)
            stats = anonymize_video(
                input_path,
                refs_dir,
                output_path,
                match_threshold=float(match_threshold),
                throttle=int(throttle),
                grace=int(grace),
                min_match_streak=int(min_match_streak),
                start_sec=start_sec,
                end_sec=end_sec,
                progress_cb=_progress,
            )
        except ValueError as exc:
            update_job(job_id, status="failed", error=str(exc), message=str(exc))
            return
        except Exception as exc:
            print(f"[anonymize-video] job {job_id} failed: {exc}")
            update_job(job_id, status="failed", error=str(exc), message="Processing failed")
            return

        if not output_path.exists():
            update_job(job_id, status="failed", error="Output video was not created")
            return

        rel_path = output_path.relative_to(REPO_ROOT / "output_data")
        static_url = f"/static/{rel_path.as_posix()}"

        if supabase is not None and user_id:
            try:
                supabase.table("profiles").upsert(
                    {
                        "id": user_id,
                        "email": f"user_{user_id[:8]}@neuroclip.local",
                        "full_name": "NeuroClip User",
                    },
                    on_conflict="id",
                ).execute()
            except Exception:
                pass
            try:
                history_query = query_note or (
                    f"ref_images={saved_refs} threshold={match_threshold}"
                )
                supabase.table("user_videos").upsert({
                    "id": job_id,
                    "user_id": user_id,
                    "title": f"Anonymized: {safe_name}",
                    "original_filename": safe_name,
                    "video_url": static_url,
                    "file_size": int(output_path.stat().st_size),
                    "duration": float(stats.get("video_duration_sec", 0)),
                    "status": "completed",
                    "metadata": {"module": "blurring"},
                }).execute()
                supabase.table("processing_history").insert({
                    "user_id": user_id,
                    "video_id": job_id,
                    "module": "blurring",
                    "input_type": "file",
                    "input_url": safe_name,
                    "query": history_query,
                    "result_url": static_url,
                    "status": "completed",
                }).execute()
            except Exception as e:
                print("Supabase persistence failed for blurring job:", e)

        update_job(
            job_id,
            status="completed",
            progress=100,
            message="Done",
            url=static_url,
            target_ids_blurred=int(stats.get("target_ids_blurred", 0)),
            processing_time_sec=float(stats.get("processing_time_sec", 0)),
            video_duration_sec=stats.get("video_duration_sec"),
            device=stats.get("device"),
        )

    start_blur_job(job_id, _run_job)

    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Video queued for anonymization. Poll /anonymize-status/{job_id}.",
        "reference_images_count": saved_refs,
    }


@app.get("/anonymize-status/{job_id}")
def anonymize_status(job_id: str):
    from blur_jobs import get_job

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8040)

@app.get("/video-embeddings")
def get_video_embeddings(video_id: Optional[str] = None, job_id: Optional[str] = None):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        if job_id:
            r = supabase.table("video_embeddings").select("*").eq("job_id", job_id).limit(1).execute()
        elif video_id:
            r = supabase.table("video_embeddings").select("*").eq("video_id", video_id).limit(1).execute()
        else:
            raise HTTPException(status_code=400, detail="Provide video_id or job_id")
        rows = r.data or []
        if not rows:
            raise HTTPException(status_code=404, detail="Embeddings not found")
        return rows[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
