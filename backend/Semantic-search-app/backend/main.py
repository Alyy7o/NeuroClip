import os
import sys
import shutil
import uuid
import json
import re
import time
import requests
import assemblyai as aai
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- Path Definitions ---
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]

# --- App Initialization ---
app = FastAPI()

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
try:
    from dotenv import load_dotenv
    # Load .env from backend root or app root
    for p in [REPO_ROOT / ".env", BASE_DIR.parent / ".env"]:
        if p.exists():
            load_dotenv(str(p))
            break
except ImportError:
    pass

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
                import torch
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
    allow_origins=[
        "http://localhost:8030",
        "http://localhost:8080",
        "http://localhost:8081",
    ], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],
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

    if not os.getenv("ASSEMBLYAI_API_KEY", "").strip():
        raise HTTPException(status_code=400, detail="ASSEMBLYAI_API_KEY not configured")
    try:
        result = generate_transcript_from_video(
            source=str(temp_path),
            output_srt_path=str(out_path) if out_path else None,
            language_code=language_code,
        )
    except ValueError as e:
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

@app.get("/health")
def health():
    """Simple health check to verify service is reachable."""
    return {"status": "ok", "service": "semantic-backend", "port": 8040}

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
    Common workflow:
    1. Transcribe (if SRT not provided)
    2. Parse SRT
    3. Generate Embeddings
    4. Save to Database (Supabase)
    """
    output_dir = REPO_ROOT / "output_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # custom_model_training is in the NeuroClip root (one level up from REPO_ROOT)
        # Try to find model in Kaggle datasets first
        model_path = REPO_ROOT.parent / "custom_model_training" / "neuroclip_v1.pth"
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            found_models = list(kaggle_input.glob("**/neuroclip_v1.pth"))
            if found_models:
                model_path = found_models[0]
        ocr_model = load_ocr_model(str(model_path))
        if ocr_model:
            frames_dir = output_dir / "frames" / job_id
            print(f"Extracting high-value frames to {frames_dir}...")
            count = extract_high_value_frames(saved_path, frames_dir, ocr_model)
            if count > 0:
                print(f"Extracted {count} frames. Running OCR...")
                ocr_text_data = run_ocr_on_frames(frames_dir)
                print(f"OCR extracted text from {len(ocr_text_data)} frames.")
    except Exception as e:
        print(f"OCR/Frame extraction failed: {e}")

    srt_out = output_dir / f"{saved_path.stem}.srt"
    
    result = {
        "id": "",
        "status": "",
        "text": "",
        "srt_path": None,
    }
    
    # 1. Transcribe or use provided SRT
    if provided_srt_path and provided_srt_path.exists():
        # Copy provided SRT to standard location
        try:
            shutil.copy2(provided_srt_path, srt_out)
            srt_content = srt_out.read_text(encoding="utf-8", errors="replace")
            result = {
                "id": "yt_caption",
                "status": "completed",
                "text": srt_content, # Approximate text
                "srt_path": srt_out
            }
        except Exception as e:
            print(f"Failed to use provided SRT: {e}")
            # Fallback to AssemblyAI?
            pass

    if not result.get("status") == "completed":
        # AssemblyAI transcription
        if not os.getenv("ASSEMBLYAI_API_KEY", "").strip():
            raise HTTPException(status_code=400, detail="ASSEMBLYAI_API_KEY not configured")
        try:
            result = generate_transcript_from_video(
                source=str(saved_path),
                output_srt_path=str(srt_out),
                language_code=None,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Transcription failed: {e}")

    # Convert SRT text to project JSON and save to output_data
    srt_text = ""
    if result.get("srt_path"):
        try:
             srt_text = Path(result["srt_path"]).read_text(encoding="utf-8", errors="replace")
        except:
             srt_text = result.get("text", "")
    elif result.get("text"):
        srt_text = result.get("text")
    
    if not srt_text:
         # Attempt to read if srt_out exists
         if srt_out.exists():
             srt_text = srt_out.read_text(encoding="utf-8", errors="replace")
    
    # If still no text, we might have an issue, but let's try parsing what we have
    if not srt_text:
        # It's possible AssemblyAI failed to generate SRT text in result but wrote to file
        pass

    sentences = parse_srt_blocks(srt_text)
    
    # Merge OCR text into sentences
    if ocr_text_data and sentences:
        for ocr_item in ocr_text_data:
            ts = ocr_item["timestamp"]
            text = ocr_item["text"]
            # Find closest sentence or append as new context
            found = False
            for s in sentences:
                s_start = float(s["starttime"])
                s_end = float(s["endtime"])
                # If OCR timestamp falls within or near the sentence, append
                if s_start <= ts <= s_end or abs(s_start - ts) < 2.0:
                    s["sentence"] += f" [On Screen: {text}]"
                    found = True
                    break
            if not found:
                # Add as a new "synthetic" sentence if no match found
                sentences.append({
                    "sentence": f"[Visual Content]: {text}",
                    "starttime": f"{ts:.2f}",
                    "endtime": f"{ts+2.0:.2f}",
                    "verbs": ["visual_ocr"]
                })
        # Keep sentences sorted by time
        sentences.sort(key=lambda x: float(x["starttime"]))

    # If parsing failed or empty, sentences is []
    
    duration_seconds = sentences[-1]["endtime"] if sentences else "0.00"
    meta = build_metadata({}, srt_path=str(saved_path.with_suffix('.srt')), video_path=str(saved_path))
    if duration_seconds and meta.get("duration") in (None, "N/A"):
        meta["duration"] = duration_seconds

    data = {"metadata": meta, "sentences": sentences}
    
    # Generate embeddings
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
        
    json_path = output_dir / f"{saved_path.stem}.v4.json"
    try:
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write JSON: {e}")

    # Persist to Supabase
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
            
            # Embeddings and vector search rows
            # We try to insert into video_embeddings (twice in original code? seems like a copy-paste error in original, but I will consolidate)
            try:
                avg_vec = None
                transcript_text = " ".join([s.get("sentence", "") for s in sentences])
                transcript_vec = None
                
                # Check if we have embeddings
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
                # print("video_embeddings insert failed:", e)
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
                    history_data = {
                        "user_id": user_id,
                        "job_id": job_id,
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
    
    # Deduplication: Check if URL was already processed
    if payload.user_id:
        try:
            # Find latest completed job for this URL
            existing = supabase.table("processing_history")\
                .select("job_id")\
                .eq("user_id", payload.user_id)\
                .eq("input_url", payload.url)\
                .eq("status", "completed")\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if existing.data and len(existing.data) > 0:
                old_job_id = existing.data[0]["job_id"]
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
                         new_job_id = old_job_id # Reuse job_id logic? Or create new? The system uses job_id as video_id.
                         # If we reuse job_id, we just point to same video.
                         try:
                             history_data = {
                                "user_id": payload.user_id,
                                "job_id": old_job_id,
                                "video_id": old_job_id,
                                "module": "summarization",
                                "input_type": "url",
                                "input_url": payload.url,
                                "query": payload.query or None,
                                "status": "completed",
                            }
                             supabase.table("processing_history").insert(history_data).execute()
                         except:
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
    
    # Configure yt-dlp - Attempt 1: With Subtitles
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' if has_ffmpeg else 'best[ext=mp4]/best',
        'outtmpl': str(uploads_dir / f"{job_id}_%(title)s.%(ext)s"),
        'noplaylist': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'convertsubtitles': 'srt',
        # 'quiet': True,
    }
    
    downloaded_video = None
    downloaded_srt = None
    
    def run_download(opts):
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(payload.url, download=True)
            return ydl.prepare_filename(info)

    try:
        try:
            downloaded_video = run_download(ydl_opts)
            # Identify SRT file if successful
            base_name = Path(downloaded_video).stem
            potential_srts = list(uploads_dir.glob(f"{base_name}*.srt"))
            if potential_srts:
                downloaded_srt = potential_srts[0]
        except Exception:
            print("yt-dlp download with subtitles failed (likely 429 or missing subs). Retrying without subtitles...")
            # Fallback: Disable subtitles and retry
            ydl_opts['writesubtitles'] = False
            ydl_opts['writeautomaticsub'] = False
            ydl_opts.pop('subtitleslangs', None)
            ydl_opts.pop('convertsubtitles', None)
            
            downloaded_video = run_download(ydl_opts)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed after retry: {e}")
    
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
    out_dir = Path(__file__).resolve().parents[2] / "backend" / "output_data"
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
    q_vec = list(map(float, get_model().encode([payload.query])[0]))
    
    # Advanced Search Logic (ported from clips_search_db)
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

    video_path = str(data.get("metadata", {}).get("file"))
    if not video_path:
        raise HTTPException(status_code=400, detail="Video path missing in metadata")
    clips_dir = out_dir / "clips" / (json_path.stem.split("_")[0])
    clips_dir.mkdir(parents=True, exist_ok=True)
    results = []
    
    for rank, (score, a, b) in enumerate(top, 1):
        # Clip Extraction Logic using start/end indices 'a' and 'b'
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
            
        # Enforce min/max duration
        dur = max(payload.min_clip_secs, end - start)
        
        # Update 'end' to reflect the minimum duration extension
        if end - start < dur:
            end = start + dur

        # Clamp to max duration
        if dur > payload.max_clip_secs:
            end = start + payload.max_clip_secs
            dur = payload.max_clip_secs
        
        # Clamp end to total video duration if available
        try:
             total = float(data.get("metadata", {}).get("duration") or 0)
             if total > 0 and end > total:
                 end = total
                 dur = max(0, end - start) # Recalculate duration if clamped
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
            cmd = [
                ff, "-y",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(dur),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(clip_path),
            ]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                ok = clip_path.exists() and clip_path.stat().st_size > 1024
            except Exception:
                ok = False
        if not ok:
            clip_path = None
        clip_rel = clip_path.relative_to(out_dir) if clip_path else None
        
        # Build text from range
        text_segment = " ".join([sentences[t].get("sentence") for t in range(a, b+1)])
        
        results.append({
            "rank": rank,
            "score": score,
            "text": text_segment,
            "start": start,
            "end": end,
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
        print(f"LLM enrichment failed in clips_search: {e}")

    return {"results": results, "count": len(results)}

class UploadAndSearchResponse(BaseModel):
    job_id: str
    json_path: str
    srt_path: str
    results: list
    count: int

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
        output_dir = Path(__file__).resolve().parents[2] / "backend" / "output_data"
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
        raise HTTPException(status_code=400, detail=ext)
    if not os.getenv("ASSEMBLYAI_API_KEY", "").strip():
        raise HTTPException(status_code=400, detail="ASSEMBLYAI_API_KEY not configured")
    srt_out = output_dir / f"{saved_path.stem}.srt"
    try:
        result = generate_transcript_from_video(
            source=str(saved_path),
            output_srt_path=str(srt_out),
            language_code=None,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    srt_text = result.get("srt_text")
    if not srt_text:
        raise HTTPException(status_code=502, detail="SRT not available")
    sentences = parse_srt_blocks(srt_text)

    # Note: upload-and-search is a faster path, but for consistency we should 
    # ideally run OCR here too if we want enhanced accuracy. 
    # However, to keep it simple, we'll focus on the primary process_video_workflow.
    # We'll add OCR here as well for full coverage.
    try:
        # Try to find model in Kaggle datasets first
        model_path = Path(__file__).resolve().parents[2] / "custom_model_training" / "neuroclip_v1.pth"
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            found_models = list(kaggle_input.glob("**/neuroclip_v1.pth"))
            if found_models:
                model_path = found_models[0]
        ocr_model = load_ocr_model(str(model_path))
        if ocr_model:
            frames_dir = output_dir / "frames" / job_id
            extract_high_value_frames(saved_path, frames_dir, ocr_model)
            ocr_text_data = run_ocr_on_frames(frames_dir)
            for ocr_item in ocr_text_data:
                ts = ocr_item["timestamp"]
                text = ocr_item["text"]
                found = False
                for s in sentences:
                    if float(s["starttime"]) <= ts <= float(s["endtime"]) or abs(float(s["starttime"]) - ts) < 2.0:
                        s["sentence"] += f" [On Screen: {text}]"
                        found = True
                        break
                if not found:
                    sentences.append({
                        "sentence": f"[Visual Content]: {text}",
                        "starttime": f"{ts:.2f}",
                        "endtime": f"{ts+2.0:.2f}",
                        "verbs": ["visual_ocr"]
                    })
            sentences.sort(key=lambda x: float(x["starttime"]))
    except Exception as e:
        print(f"OCR in upload-and-search failed: {e}")

    duration_seconds = sentences[-1]["endtime"] if sentences else "0.00"
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
    except Exception:
        pass
    # Persist to supabase
    try:
        if supabase is not None:
            if user_id:
                meta_payload = {
                    "id": job_id,
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
                    supabase.table("processing_history").insert({
                        "user_id": user_id,
                        "job_id": job_id,
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
    return UploadAndSearchResponse(job_id=job_id, json_path=str(json_path), srt_path=str(srt_out), results=r["results"], count=r["count"]) 

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

# --- LLM Refinement via Google Gemini ---
_gemini_model = None

def get_gemini_model():
    global _gemini_model
    if _gemini_model is None:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            _gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        except Exception as e:
            print(f"Gemini init failed: {e}")
            _gemini_model = None
    return _gemini_model

def refine_with_llm(query: str, candidates: list) -> list:
    """
    Send candidate transcript segments to Gemini LLM for refinement.
    Each candidate: {"index": int, "text": str, "start": float, "end": float}
    Returns list of {"index": int, "start": float, "end": float, "summary": str}
    Falls back to empty list on any failure.
    """
    model = get_gemini_model()
    if model is None:
        return []
    
    segments_text = ""
    for c in candidates:
        segments_text += f"[Segment {c['index']}] {c['start']:.1f}s - {c['end']:.1f}s: \"{c['text']}\"\n"
    
    prompt = f"""You are a video content analyst. Given a user's query and transcript segments with timestamps, identify which segments best discuss the queried topic.

For each relevant segment, return:
- The segment index (matching the input)
- The exact start and end timestamps from the segment
- A 2-3 sentence summary of what is discussed about the topic in that segment

User Query: "{query}"

Transcript Segments:
{segments_text}

Respond ONLY with valid JSON in this exact format, no markdown code fences:
{{{{
  "results": [
    {{{{
      "segment_index": 0,
      "start": 12.5,
      "end": 45.3,
      "summary": "The speaker discusses..."
    }}}}
  ]
}}}}"""
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3].strip()
        if text.startswith("json"):
            text = text[4:].strip()
        
        parsed = json.loads(text)
        return parsed.get("results", [])
    except Exception as e:
        print(f"LLM refinement failed: {e}")
        return []


@app.post("/clips/search-db")
def clips_search_db(payload: DbSearchRequest):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        # Locate JSON by job_id instead of user_videos
        out_dir = Path(__file__).resolve().parents[2] / "backend" / "output_data"
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
    # 5. Build FFmpeg Command
    #
    #    CRITICAL FIX: All quality/encoding flags MUST appear before the output
    #    path. Previously flags were appended via cmd.extend() AFTER the output
    #    path string, causing FFmpeg to silently ignore them — resulting in
    #    default (very high quality / large file) encoding on large videos.
    #
    #    Scale filter: "scale=-2:min(720,ih)" caps height at 720p while
    #    preserving aspect ratio. If the video is already smaller than 720p it
    #    is left untouched (ih = input height).
    # -------------------------------------------------------------------------
    vf = "scale=-2:min(720\\,ih)"   # cap at 720p; keep smaller videos as-is

    # --- common input flags ---
    cmd = [
        ff, "-y",
        "-i", str(input_path),
        "-vf", vf,
    ]

    if has_gpu:
        # GPU path: HEVC NVENC — VBR with tighter bitrate ceiling
        cmd += [
            "-c:v",     "hevc_nvenc",
            "-preset",  "p4",        # better compression than p3
            "-rc",      "vbr",
            "-cq",      "28",
            "-b:v",     "1.5M",      # target bitrate  (was 2M)
            "-maxrate", "2.5M",      # hard cap        (was 3M)
            "-bufsize", "5M",
            "-qmin",    "24",
            "-qmax",    "36",        # allow more compression on easy scenes
        ]
        mode_label = "HEVC_NVENC"
    else:
        # CPU path: libx265 — CRF only (no conflicting -b:v)
        # Mixing -crf with -b:v in libx265 causes undefined/bloated output.
        cmd += [
            "-c:v",        "libx265",
            "-preset",     "medium",          # better ratio than 'fast'
            "-crf",        "28",              # sole quality knob — no -b:v
            "-x265-params","pools=4:frame-threads=2",  # explicit Kaggle-safe threading
        ]
        mode_label = "LIBX265"

    # --- shared audio + container flags, then output path (always last) ---
    cmd += [
        "-c:a",      "aac",
        "-b:a",      "96k",          # reduced from 128k
        "-movflags", "+faststart",
        str(output_path),            # OUTPUT PATH IS LAST — never append flags after this
    ]

    # -------------------------------------------------------------------------
    # 6. Run Compression
    # -------------------------------------------------------------------------
    start_time = time.time()
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        err_out = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
        raise HTTPException(status_code=500, detail=f"Compression failed: {err_out}")

    duration = time.time() - start_time

    # -------------------------------------------------------------------------
    # 7. Validate Output
    # -------------------------------------------------------------------------
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise HTTPException(status_code=500, detail="Compression resulted in empty file")

    compressed_size = output_path.stat().st_size
    reduction = (
        100 - (compressed_size / original_size * 100)
        if original_size > 0 else 0
    )

    # -------------------------------------------------------------------------
    # 8. Save to Processing History & Video Metadata
    # -------------------------------------------------------------------------
    if supabase is not None:
        try:
            rel_path = output_path.relative_to(REPO_ROOT / "output_data")
            static_url = f"/static/{rel_path.as_posix()}"

            # A. processing_history (The job log)
            if user_id:
                supabase.table("processing_history").insert({
                    "user_id":    user_id,
                    "job_id":     job_id,
                    "video_id":   job_id,
                    "module":     "compression",
                    "input_type": "file",
                    "input_url":  safe_name,
                    "query":      (
                        f"H.265 / {mode_label} / 720p / "
                        f"Total Proc: {duration:.2f}s / "
                        f"Reduction: {reduction:.1f}%"
                    ),
                    "status": "completed",
                }).execute()
            
            # B. user_videos (Primary metadata table for all video types)
            # Storing here ensures visual tracking in history even for non-AI tasks
            if user_id:
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
                        "encoder": mode_label,
                        "original_size": original_size,
                        "reduction": reduction,
                        "proc_time": duration
                    }
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
    }


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
