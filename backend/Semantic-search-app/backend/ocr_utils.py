import os
import cv2
import shutil
import subprocess
from pathlib import Path

# ====================
# CONFIGURATION
# ====================
SAMPLE_INTERVAL = 5.0   # default seconds between sampled frames
MAX_FRAMES = 1000       # cap total OCR frames to avoid spending hours on OCR
MIN_TEXT_LENGTH = 4     # minimum characters to keep OCR result

# Adaptive sampling thresholds (duration_seconds -> sample_interval)
# For very long videos, sample less frequently to stay under MAX_FRAMES
ADAPTIVE_INTERVALS = [
    (7200,  30.0),   # > 2 hours  → every 30s
    (3600,  20.0),   # > 1 hour   → every 20s
    (1800,  10.0),   # > 30 min   → every 10s
    (0,      5.0),   # default    → every 5s
]

# Videos longer than this use FFmpeg directly (OpenCV seeking is too slow on large H.264)
FFMPEG_THRESHOLD_SECONDS = 1800  # 30 minutes

print(f"--- [NeuroClip OCR] Direct EasyOCR mode (adaptive interval, max {MAX_FRAMES} frames) ---")

# ====================
# HELPERS
# ====================

def _get_adaptive_interval(duration_seconds: float) -> float:
    """Pick a sample interval based on video duration to keep frame count manageable."""
    for threshold, interval in ADAPTIVE_INTERVALS:
        if duration_seconds > threshold:
            return interval
    return SAMPLE_INTERVAL


def _get_video_duration(video_path) -> float:
    """Get video duration using ffprobe (reliable) or OpenCV (fallback)."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        try:
            result = subprocess.run(
                [ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass

    # Fallback: use OpenCV to get duration
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total > 0 and fps > 0:
            return total / fps
    return 0.0


# ====================
# FRAME EXTRACTION
# ====================

def _extract_frames_ffmpeg(video_path, output_dir, sample_interval=SAMPLE_INTERVAL):
    """
    Frame extraction using FFmpeg subprocess.
    Works with ANY codec (AV1, VP9, H.265, H.264, etc.) and handles
    long videos efficiently via sequential decoding.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("[OCR] FFmpeg not found on PATH — cannot extract frames")
        return 0

    fps_filter = f"fps=1/{sample_interval}"
    # Use %05d numbering; we'll rename to timestamp names after
    output_pattern = str(output_dir / "ffout_%05d.jpg")

    cmd = [
        ffmpeg_bin,
        "-i", str(video_path),
        "-vf", fps_filter,
        "-q:v", "2",
        "-y",
        output_pattern,
    ]

    # Scale timeout with video duration (at least 5 min, up to 30 min)
    duration = _get_video_duration(video_path)
    timeout = max(300, min(1800, int(duration / 10)))

    print(f"[OCR] FFmpeg: extracting 1 frame every {sample_interval}s "
          f"(~{int(duration / sample_interval)} frames, timeout={timeout}s)...")

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        if proc.returncode != 0:
            stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-500:]
            print(f"[OCR] FFmpeg exited with code {proc.returncode}: ...{stderr_tail}")
    except subprocess.TimeoutExpired:
        print(f"[OCR] FFmpeg frame extraction timed out (>{timeout}s)")
        # Still check if any frames were produced before timeout
    except Exception as e:
        print(f"[OCR] FFmpeg frame extraction error: {e}")
        return 0

    # Rename sequential output to timestamp-based filenames
    raw_files = sorted(output_dir.glob("ffout_*.jpg"))
    if not raw_files:
        print("[OCR] FFmpeg produced no output frames")
        return 0

    renamed = 0
    for idx, f in enumerate(raw_files):
        timestamp_s = int(idx * sample_interval)
        new_name = f"frame_{timestamp_s:05d}s.jpg"
        new_path = output_dir / new_name
        try:
            f.rename(new_path)
        except Exception:
            pass
        renamed += 1

    print(f"[OCR] FFmpeg extracted {renamed} frames successfully")
    return renamed


def extract_all_video_frames(video_path, output_dir, sample_interval=None):
    """
    Extract frames from a video at regular intervals.
    
    Strategy:
      - Auto-compute sample interval based on video duration
      - For long videos (>30 min): use FFmpeg directly (OpenCV seeking is too slow)
      - For short videos: try OpenCV first, fall back to FFmpeg
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        sample_interval: Override seconds between captures (auto if None)
    
    Returns:
        Number of frames extracted
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get duration to make smart decisions
    duration = _get_video_duration(video_path)
    
    # Auto-compute adaptive interval
    if sample_interval is None:
        sample_interval = _get_adaptive_interval(duration)
    
    estimated_frames = int(duration / sample_interval) if duration > 0 else 0
    print(f"[OCR] Video duration: {duration:.0f}s ({duration/3600:.1f}h), "
          f"interval: {sample_interval}s, estimated frames: {estimated_frames}")

    # For long videos, go straight to FFmpeg — OpenCV seeking is unreliable/slow
    if duration > FFMPEG_THRESHOLD_SECONDS:
        print(f"[OCR] Long video (>{FFMPEG_THRESHOLD_SECONDS}s) — using FFmpeg directly "
              f"(OpenCV frame-seeking is too slow for large files)")
        return _extract_frames_ffmpeg(video_path, output_dir, sample_interval)

    # For shorter videos, try OpenCV first
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[OCR] OpenCV could not open video — trying FFmpeg")
        return _extract_frames_ffmpeg(video_path, output_dir, sample_interval)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 24.0
    
    frame_interval = int(fps * sample_interval)
    if frame_interval < 1:
        frame_interval = 1
    
    print(f"[OCR] OpenCV: {total_frames} frames at {fps:.1f}fps, "
          f"sampling every {frame_interval} frames")
    
    current_frame = 0
    saved_count = 0
    
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            if saved_count == 0:
                print(f"[OCR] OpenCV failed to read first frame — falling back to FFmpeg")
                break
            break
        
        timestamp_s = current_frame / fps
        filename = f"frame_{int(timestamp_s):05d}s.jpg"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), frame)
        saved_count += 1
        
        current_frame += frame_interval
    
    cap.release()
    
    if saved_count == 0:
        return _extract_frames_ffmpeg(video_path, output_dir, sample_interval)
    
    print(f"[OCR] Extracted {saved_count} frames via OpenCV ({duration:.1f}s video)")
    return saved_count


def run_ocr_on_frames(image_folder, min_text_length=MIN_TEXT_LENGTH):
    """
    Runs EasyOCR on all images in image_folder.
    Returns list of dicts: [{'timestamp': float, 'text': str}]
    Only keeps frames where meaningful text was detected.
    """
    image_folder = Path(image_folder)
    if not image_folder.exists():
        print(f"[OCR] Frame folder not found: {image_folder}")
        return []

    try:
        import easyocr
        # Check for GPU
        use_gpu = False
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            pass
        reader = easyocr.Reader(['en'], gpu=use_gpu)
        print(f"[OCR] EasyOCR initialized (GPU: {use_gpu})")
    except ImportError:
        print("[OCR WARNING] EasyOCR not installed. Install with: pip install easyocr")
        return []
    except Exception as e:
        print(f"[OCR ERROR] EasyOCR initialization failed: {e}")
        return []

    files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    if not files:
        print(f"[OCR] No JPG frames found in {image_folder}")
        return []
    
    print(f"[OCR] Running OCR on {len(files)} frames...")
    results = []
    frames_with_text = 0
    
    for i, filename in enumerate(files):
        try:
            # Extract timestamp from filename: frame_00045s.jpg -> 45
            ts_str = filename.replace("frame_", "").replace("slide_", "").replace("s.jpg", "")
            seconds = int(ts_str)
        except (ValueError, IndexError):
            seconds = 0

        path = image_folder / filename
        try:
            ocr_result = reader.readtext(str(path), detail=0)
        except Exception as e:
            print(f"[OCR] Failed on {filename}: {e}")
            continue
        
        # Filter short junk text (< min_text_length chars)
        clean_text = [text.strip() for text in ocr_result if len(text.strip()) >= min_text_length]
        
        if clean_text:
            full_content = " ".join(clean_text)
            results.append({
                "timestamp": float(seconds),
                "text": full_content
            })
            frames_with_text += 1
        
        # Progress logging every 20 frames
        if (i + 1) % 20 == 0:
            print(f"[OCR] Progress: {i+1}/{len(files)} frames processed, {frames_with_text} with text")
    
    print(f"[OCR] Complete: {frames_with_text}/{len(files)} frames contained text ({len(results)} results)")
    return results


# ====================
# LEGACY COMPATIBILITY
# ====================
# Keep old function signatures so existing code doesn't break

def load_ocr_model(path=None):
    """
    Legacy wrapper — no longer needed since we use direct EasyOCR.
    Returns a sentinel value so callers know OCR is available.
    """
    # Just check if EasyOCR is importable
    try:
        import easyocr
        print("[OCR] EasyOCR available — direct OCR mode active")
        return True  # Return truthy value so callers proceed
    except ImportError:
        print("[OCR WARNING] EasyOCR not installed")
        return None


def extract_high_value_frames(video_path, output_dir, model=None, threshold=None):
    """
    Legacy wrapper — now extracts ALL frames uniformly instead of model-filtered ones.
    """
    return extract_all_video_frames(video_path, output_dir, sample_interval=SAMPLE_INTERVAL)
