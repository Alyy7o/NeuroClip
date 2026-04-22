import os
import cv2
import shutil
import subprocess
from pathlib import Path

# ====================
# CONFIGURATION
# ====================
# Frame sampling: extract 1 frame every SAMPLE_INTERVAL seconds
SAMPLE_INTERVAL = 5.0  # seconds between sampled frames (5s = good coverage with 40% less compute)
MIN_TEXT_LENGTH = 4     # minimum characters to keep OCR result

print(f"--- [NeuroClip OCR] Direct EasyOCR mode (every {SAMPLE_INTERVAL}s) ---")

# ====================
# FRAME EXTRACTION
# ====================

def _extract_frames_ffmpeg(video_path, output_dir, sample_interval=SAMPLE_INTERVAL):
    """
    Fallback frame extraction using FFmpeg subprocess.
    Works with ANY codec (AV1, VP9, H.265, etc.) that FFmpeg supports.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("[OCR] FFmpeg not found on PATH — cannot use FFmpeg fallback")
        return 0

    fps_filter = f"fps=1/{sample_interval}"
    output_pattern = str(output_dir / "frame_%05ds.jpg")

    cmd = [
        ffmpeg_bin,
        "-i", str(video_path),
        "-vf", fps_filter,
        "-frame_pts", "1",
        "-q:v", "2",
        "-y",
        output_pattern,
    ]

    print(f"[OCR] FFmpeg fallback: extracting frames with filter '{fps_filter}'...")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,  # 10 min timeout for very long videos
        )
        if proc.returncode != 0:
            stderr_tail = proc.stderr.decode("utf-8", errors="replace")[-500:]
            print(f"[OCR] FFmpeg exited with code {proc.returncode}: {stderr_tail}")
    except subprocess.TimeoutExpired:
        print("[OCR] FFmpeg frame extraction timed out (>600s)")
        return 0
    except Exception as e:
        print(f"[OCR] FFmpeg frame extraction error: {e}")
        return 0

    # FFmpeg names frames sequentially (frame_00001s.jpg, frame_00002s.jpg, ...)
    # We need to rename them to match our timestamp convention
    raw_files = sorted(output_dir.glob("frame_*.jpg"))
    if not raw_files:
        print("[OCR] FFmpeg produced no output frames")
        return 0

    # Rename to timestamp-based names
    renamed = 0
    for idx, f in enumerate(raw_files):
        timestamp_s = int(idx * sample_interval)
        new_name = f"frame_{timestamp_s:05d}s.jpg"
        new_path = output_dir / new_name
        if f.name != new_name:
            try:
                f.rename(new_path)
            except Exception:
                pass  # Keep original name if rename fails
        renamed += 1

    print(f"[OCR] FFmpeg extracted {renamed} frames successfully")
    return renamed


def extract_all_video_frames(video_path, output_dir, sample_interval=SAMPLE_INTERVAL):
    """
    Extract frames from the ENTIRE video at regular intervals.
    Strategy:
      1. Try OpenCV (fast, but fails on AV1/VP9 codecs)
      2. If OpenCV gets 0 frames, fall back to FFmpeg subprocess
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        sample_interval: Seconds between frame captures (default 5.0)
    
    Returns:
        Number of frames extracted
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[OCR] OpenCV could not open video — trying FFmpeg fallback")
        return _extract_frames_ffmpeg(video_path, output_dir, sample_interval)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 24.0
    
    duration = total_frames / fps
    frame_interval = int(fps * sample_interval)
    if frame_interval < 1:
        frame_interval = 1
    
    print(f"[OCR] Video: {duration:.1f}s, {fps:.1f}fps, {total_frames} total frames")
    print(f"[OCR] Sampling 1 frame every {sample_interval}s ({total_frames // frame_interval} frames to process)")
    
    current_frame = 0
    saved_count = 0
    
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            # If we haven't saved any frames yet and OpenCV can't read,
            # this is likely a codec issue — break early to try FFmpeg
            if saved_count == 0:
                print(f"[OCR] OpenCV failed to read first frame (likely unsupported codec: AV1/VP9)")
                break
            # Otherwise, we've just reached the end of the video
            break
        
        timestamp_s = current_frame / fps
        filename = f"frame_{int(timestamp_s):05d}s.jpg"
        save_path = output_dir / filename
        cv2.imwrite(str(save_path), frame)
        saved_count += 1
        
        current_frame += frame_interval
    
    cap.release()
    
    # If OpenCV extracted 0 frames, fall back to FFmpeg
    if saved_count == 0:
        print(f"[OCR] OpenCV extracted 0 frames — falling back to FFmpeg")
        return _extract_frames_ffmpeg(video_path, output_dir, sample_interval)
    
    print(f"[OCR] Extracted {saved_count} frames from video ({duration:.1f}s)")
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
