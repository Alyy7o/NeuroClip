"""
Phase 3: dual-GPU (or multi-GPU) chunk-parallel blur for long videos.

Splits the source into time chunks, runs BlurEngine per GPU in parallel,
concatenates video, then muxes original audio once.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _ffmpeg() -> Optional[str]:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ffprobe_duration(path: Path) -> float:
    ff = shutil.which("ffprobe") or _ffmpeg()
    if not ff:
        return 0.0
    # Prefer ffprobe; fall back to OpenCV if needed
    probe = shutil.which("ffprobe")
    if probe:
        try:
            out = subprocess.check_output(
                [
                    probe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                text=True,
                timeout=60,
            )
            return max(0.0, float(out.strip()))
        except Exception:
            pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        return float(n / max(fps, 1.0))
    except Exception:
        return 0.0


def _gpu_count() -> int:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
    except Exception:
        pass
    return 0


def _split_chunks(
    video: Path, work: Path, n: int, duration: float
) -> List[Tuple[Path, float, float]]:
    """Return list of (chunk_path, start_sec, dur_sec)."""
    ff = _ffmpeg()
    if not ff or duration <= 0 or n < 2:
        return [(video, 0.0, duration)]

    chunk_dur = duration / n
    chunks: List[Tuple[Path, float, float]] = []
    for i in range(n):
        start = i * chunk_dur
        dur = chunk_dur if i < n - 1 else max(0.1, duration - start)
        out = work / f"chunk_{i:02d}.mp4"
        # Re-encode lightly so keyframes align (copy can miss -ss accuracy)
        cmd = [
            ff,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{dur:.3f}",
            "-i",
            str(video),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-an",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0 or not out.exists():
            logger.warning("[blur-chunk] split failed for chunk %d: %s", i, proc.stderr[-400:])
            return [(video, 0.0, duration)]
        chunks.append((out, start, dur))
    return chunks


def _concat_list_entry(path: Path) -> str:
    """FFmpeg concat demuxer line; escape quotes; prefer forward slashes."""
    s = path.resolve().as_posix().replace("'", r"'\''")
    return f"file '{s}'\n"


def _concat_videos(parts: List[Path], out: Path) -> bool:
    ff = _ffmpeg()
    if not ff or not parts:
        return False
    if any(not p.exists() for p in parts):
        logger.warning("[blur-chunk] concat missing part(s): %s", parts)
        return False
    lst = out.with_suffix(".txt")
    lst.write_text("".join(_concat_list_entry(p) for p in parts), encoding="utf-8")
    cmd = [
        ff,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(lst),
        "-c",
        "copy",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    try:
        lst.unlink(missing_ok=True)
    except Exception:
        pass
    return proc.returncode == 0 and out.exists()


def _mux_audio(video_no_audio: Path, audio_src: Path, dst: Path) -> bool:
    from blur_service import _transcode_for_browser

    return _transcode_for_browser(video_no_audio, dst, audio_src=audio_src)


def _process_one_chunk(
    chunk_path: Path,
    ref_dir: Path,
    out_path: Path,
    device: str,
    match_threshold: float,
    throttle: int,
    grace: int,
    min_match_streak: int,
) -> Dict:
    # Isolate device for this worker; never nest chunk-parallel inside a chunk
    os.environ["BLUR_DEVICE"] = device
    os.environ["BLUR_CHUNK_PARALLEL"] = "0"
    # Keep parent require-gpu policy if set; default off for local CPU tests
    if "BLUR_REQUIRE_GPU" not in os.environ:
        os.environ["BLUR_REQUIRE_GPU"] = "0"
    from blur_service import BlurEngine

    engine = BlurEngine(device=device)
    masters, master_identity = engine.build_master_signatures(ref_dir)
    stats = engine.process_video(
        chunk_path,
        out_path,
        masters,
        master_identity,
        match_threshold=match_threshold,
        throttle=throttle,
        grace=grace,
        min_match_streak=min_match_streak,
    )
    if not out_path.exists():
        raise RuntimeError(f"chunk output missing after process_video: {out_path}")
    stats["device"] = device
    stats["chunk"] = str(chunk_path)
    return stats


def should_use_chunk_parallel(video_path: Path) -> bool:
    raw = os.getenv("BLUR_CHUNK_PARALLEL", "").strip().lower()
    gpus = _gpu_count()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return gpus >= 2
    # Auto: multi-GPU and video longer than 3 minutes
    if gpus < 2:
        return False
    dur = _ffprobe_duration(video_path)
    min_sec = float(os.getenv("BLUR_CHUNK_MIN_SEC", "180"))
    return dur >= min_sec


def anonymize_video_chunked(
    video_path: Path,
    reference_dir: Path,
    output_path: Path,
    *,
    match_threshold: float = 0.78,
    throttle: int = 3,
    grace: int = 30,
    min_match_streak: int = 2,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict:
    t0 = time.perf_counter()
    gpus = _gpu_count()
    if gpus < 2:
        raise RuntimeError("chunk parallel requires >= 2 GPUs")
    n_chunks = min(
        max(2, int(os.getenv("BLUR_CHUNKS", str(min(2, gpus))))),
        gpus,
        4,
    )
    duration = _ffprobe_duration(video_path)
    if duration <= 0:
        raise ValueError(f"Cannot determine duration for chunking: {video_path}")

    work = Path(tempfile.mkdtemp(prefix="neuroclip_blur_chunks_"))
    try:
        if progress_cb:
            progress_cb(1.0, "Splitting video for multi-GPU")
        chunks = _split_chunks(video_path, work, n_chunks, duration)
        if len(chunks) < 2:
            raise RuntimeError("chunk split fell back to single file")

        jobs = []
        for i, (cpath, _start, _dur) in enumerate(chunks):
            device = f"cuda:{i % gpus}"
            out_c = work / f"out_{i:02d}.mp4"
            jobs.append(
                (
                    cpath,
                    reference_dir,
                    out_c,
                    device,
                    match_threshold,
                    throttle,
                    grace,
                    min_match_streak,
                )
            )

        if progress_cb:
            progress_cb(5.0, f"Processing {len(jobs)} chunks on {gpus} GPU(s)")

        results: List[Dict] = []
        outs: List[Path] = []
        # Separate processes: ORT/YOLO sessions are not thread-safe across GPUs
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            futs = {
                ex.submit(_process_one_chunk, *job): job[2] for job in jobs
            }
            done_n = 0
            for fut in as_completed(futs):
                out_path = futs[fut]
                stats = fut.result()
                results.append(stats)
                outs.append(out_path)
                done_n += 1
                if progress_cb:
                    pct = 5.0 + 85.0 * (done_n / len(jobs))
                    progress_cb(pct, f"Chunk {done_n}/{len(jobs)} done")

        # Order outputs by name (chunk_00, chunk_01, …)
        outs = sorted(outs, key=lambda p: p.name)
        merged_raw = work / "merged_raw.mp4"
        if progress_cb:
            progress_cb(92.0, "Concatenating chunks")
        if not _concat_videos(outs, merged_raw):
            raise RuntimeError("Failed to concat blurred chunks")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if progress_cb:
            progress_cb(95.0, "Muxing audio")
        if not _mux_audio(merged_raw, video_path, output_path):
            # Fall back to copy merged without audio remux
            shutil.copy2(merged_raw, output_path)

        id_ever = max((int(r.get("identities_ever_locked") or 0) for r in results), default=0)
        elapsed = time.perf_counter() - t0
        if progress_cb:
            progress_cb(100.0, "Done")
        return {
            "target_ids_blurred": id_ever,
            "identities_ever_locked": id_ever,
            "num_identities": max(
                (int(r.get("num_identities") or 0) for r in results), default=0
            ),
            "chunks": len(jobs),
            "gpus": gpus,
            "chunk_stats": results,
            "processing_time_sec": round(elapsed, 2),
            "output_path": str(output_path),
            "browser_ready": output_path.exists(),
            "chunk_parallel": True,
        }
    finally:
        try:
            shutil.rmtree(work, ignore_errors=True)
        except Exception:
            pass
