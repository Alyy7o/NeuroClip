"""
Selective, reference-guided face anonymization for NeuroClip.

Research claim (frozen — see docs/Blurring_Module_Paper_Roadmap.md §0):
  Blur ONLY faces that match user-provided reference photo(s) in any video.
  Not NLP-driven, not plates/IDs, not blur-everyone.

Pipeline:
  1. One master embedding per reference image (InsightFace buffalo_l).
  2. YOLOv8n person track (BoT-SORT) + SCRFD face boxes on person crops.
  3. Max-sim match to any master; multi-lock + re-attach after ID switch.
  4. Adaptive elliptical Gaussian (or mosaic) blur; FFmpeg H.264 for browser.

Env:
  BLUR_DEVICE, BLUR_REQUIRE_GPU, BLUR_YOLO_WEIGHTS, BLUR_TRACKER
  BLUR_MATCH_THRESHOLD (default 0.78), BLUR_FACE_DETECT, BLUR_MIN_FACE_PX
  BLUR_MAX_THROTTLE (default 4), BLUR_OPERATOR=gaussian|mosaic
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROCESS_LOCK = threading.Lock()
_ENGINE: Optional["BlurEngine"] = None
DEFAULT_MATCH_THRESHOLD = float(os.getenv("BLUR_MATCH_THRESHOLD", "0.78"))
DEFAULT_MIN_FACE_PX = int(os.getenv("BLUR_MIN_FACE_PX", "28"))
DEFAULT_FACE_DETECT = os.getenv("BLUR_FACE_DETECT", "insightface").strip().lower()
DEFAULT_MAX_THROTTLE = int(os.getenv("BLUR_MAX_THROTTLE", "4"))
DEFAULT_OPERATOR = os.getenv("BLUR_OPERATOR", "gaussian").strip().lower()


class FaceHit(NamedTuple):
    """Face box in full-frame coords plus optional embedding from the same detect call."""

    box: Tuple[int, int, int, int]
    embedding: Optional[np.ndarray]
    det_score: float
    yaw: float = 0.0  # InsightFace pose yaw (deg); |yaw| high ⇒ profile/side view


def _face_yaw_deg(face) -> float:
    """Best-effort yaw from InsightFace face object (degrees)."""
    pose = getattr(face, "pose", None)
    if pose is not None:
        try:
            arr = np.asarray(pose, dtype=np.float32).ravel()
            if arr.size >= 2:
                return float(arr[1])  # pitch, yaw, roll — yaw is index 1 in insightface
            if arr.size == 1:
                return float(arr[0])
        except Exception:
            pass
    yaw = getattr(face, "yaw", None)
    if yaw is not None:
        try:
            return float(yaw)
        except Exception:
            pass
    return 0.0


def _effective_match_threshold(base: float, yaw_deg: float) -> float:
    """Slightly relax cosine threshold for profile faces (keep floor high to limit FPs)."""
    yaw = abs(float(yaw_deg))
    delta = float(os.getenv("BLUR_PROFILE_THRESHOLD_DELTA", "0.06"))
    floor = float(os.getenv("BLUR_MATCH_FLOOR", "0.70"))
    if yaw >= 45:
        return max(floor, base - delta)
    if yaw >= 25:
        return max(floor, base - 0.5 * delta)
    return base


def _smooth_box(
    prev: Optional[Tuple[int, int, int, int]],
    new: Tuple[int, int, int, int],
    alpha: float = 0.35,
) -> Tuple[int, int, int, int]:
    """EMA box smoothing — kills frame-to-frame blur jitter."""
    if prev is None:
        return new
    a = float(np.clip(alpha, 0.05, 1.0))
    out = []
    for p, n in zip(prev, new):
        out.append(int(round((1.0 - a) * p + a * n)))
    x1, y1, x2, y2 = out
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return new
    return (x1, y1, x2, y2)


def _face_belongs_to_person(
    face_box: Tuple[int, int, int, int],
    person_box: Tuple[int, int, int, int],
) -> float:
    """
    Association score in [0, 1+]. Negative ⇒ reject.
    Prevents a frontal neighbor face from being stolen into another person's crop.
    """
    px1, py1, px2, py2 = person_box
    fx1, fy1, fx2, fy2 = face_box
    pw, ph = max(1, px2 - px1), max(1, py2 - py1)
    cx = (fx1 + fx2) / 2.0
    cy = (fy1 + fy2) / 2.0

    # Face center must sit in the head band (upper ~62% of person)
    if cy > py1 + 0.62 * ph or cy < py1 - 0.05 * ph:
        return -1.0
    margin_x = 0.06 * pw
    if cx < px1 - margin_x or cx > px2 + margin_x:
        return -1.0

    # Inside-person bonus
    inside = 1.0 if (px1 <= cx <= px2 and py1 <= cy <= py2) else 0.45
    iou = _box_iou(face_box, person_box)
    if iou < 0.02 and inside < 1.0:
        return -1.0

    # Prefer faces higher in the torso (true head)
    head_y = 1.0 - min(1.0, max(0.0, (cy - py1) / max(ph * 0.62, 1.0)))
    return inside * (0.55 + 0.45 * head_y) * (0.4 + 0.6 * min(1.0, iou * 3.0))


def _on_kaggle() -> bool:
    return bool(
        os.getenv("KAGGLE_KERNEL_RUN_TYPE")
        or os.getenv("KAGGLE_URL_BASE")
        or Path("/kaggle").exists()
    )


def _require_gpu() -> bool:
    raw = os.getenv("BLUR_REQUIRE_GPU", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return _on_kaggle()


def _resolve_ffmpeg() -> Optional[str]:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_for_browser(
    src: Path,
    dst: Path,
    *,
    audio_src: Optional[Path] = None,
    audio_start_sec: Optional[float] = None,
    audio_end_sec: Optional[float] = None,
) -> bool:
    """
    H.264 for HTML5 playback. OpenCV raw has no audio — mux audio from the
    original input when available.
    """
    ff = _resolve_ffmpeg()
    if not ff or not src.exists():
        return False

    timeout_sec = int(os.getenv("BLUR_FFMPEG_TIMEOUT", "1800"))
    has_audio = bool(audio_src and Path(audio_src).exists())

    def _run(cmd: List[str]) -> bool:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            if proc.returncode != 0:
                logger.warning(
                    "[blur] FFmpeg failed (rc=%s): %s",
                    proc.returncode,
                    (proc.stderr or "")[-800:],
                )
            return proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0
        except Exception as exc:
            logger.warning("[blur] FFmpeg transcode failed: %s", exc)
            return False

    video_encode = [
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]

    if has_audio:
        # Video from processed raw; audio from original (optional trim).
        cmd: List[str] = [ff, "-y", "-i", str(src)]
        if audio_start_sec is not None and audio_start_sec > 0:
            cmd += ["-ss", f"{float(audio_start_sec):.3f}"]
        cmd += ["-i", str(audio_src)]
        if audio_end_sec is not None and audio_end_sec > 0:
            start = float(audio_start_sec or 0.0)
            dur = max(0.1, float(audio_end_sec) - start)
            cmd += ["-t", f"{dur:.3f}"]
        cmd += [
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
            *video_encode,
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(dst),
        ]
        if _run(cmd):
            logger.info("[blur] Encoded with original audio muxed from %s", audio_src)
            return True
        logger.warning("[blur] Audio mux failed; retrying video-only encode")

    # Video-only fallback (silent)
    cmd = [ff, "-y", "-i", str(src), *video_encode, "-an", str(dst)]
    return _run(cmd)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    if a.shape[0] != b.shape[0]:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _max_sim_to_masters(emb: np.ndarray, masters: np.ndarray) -> Tuple[float, int]:
    """Return (best_sim, master_index). masters shape (N, D)."""
    best_sim = -1.0
    best_i = 0
    for i in range(masters.shape[0]):
        sim = _cosine_similarity(masters[i], emb)
        if sim > best_sim:
            best_sim = sim
            best_i = i
    return best_sim, best_i


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_center_dist(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _odd_clamp(v: int, lo: int, hi: int) -> int:
    v = max(lo, min(hi, int(v)))
    if v % 2 == 0:
        v += 1
    return max(lo, min(hi, v))


def _apply_face_anonymize(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    operator: str = "gaussian",
) -> None:
    """Adaptive elliptical blur or mosaic on face ROI."""
    h, w = frame.shape[:2]
    pad = int(0.08 * max(x2 - x1, y2 - y1, 1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    rh, rw = roi.shape[:2]
    k = _odd_clamp(int(0.45 * max(rw, rh)), 15, 99)
    mask = np.zeros((rh, rw), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (rw // 2, rh // 2),
        (max(1, rw // 2), max(1, rh // 2)),
        0,
        0,
        360,
        255,
        -1,
    )
    if operator == "mosaic":
        block = max(4, min(rw, rh) // 12)
        small = cv2.resize(
            roi,
            (max(1, rw // block), max(1, rh // block)),
            interpolation=cv2.INTER_LINEAR,
        )
        pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        anonymized = pix
    else:
        anonymized = cv2.GaussianBlur(roi, (k, k), 0)

    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None]
    blended = (anonymized.astype(np.float32) * mask_f + roi.astype(np.float32) * (1.0 - mask_f)).astype(
        np.uint8
    )
    frame[y1:y2, x1:x2] = blended


class BlurEngine:
    """Lazy-loaded YOLO + InsightFace for reference-based face blurring."""

    def __init__(self) -> None:
        self._cuda_available = False
        self._cuda_name = ""
        self._ort_providers: List[str] = []
        self._insightface_on_gpu = False
        self._device = os.getenv("BLUR_DEVICE", "").strip()

        try:
            import torch

            self._cuda_available = bool(torch.cuda.is_available())
            if self._cuda_available:
                self._cuda_name = torch.cuda.get_device_name(0)
        except Exception:
            self._cuda_available = False

        if not self._device:
            self._device = "cuda:0" if self._cuda_available else "cpu"

        self._yolo = None
        self._face_yolo = None
        self._insightface = None
        self._haar_cascade: Optional[cv2.CascadeClassifier] = None
        self._face_detect_mode = DEFAULT_FACE_DETECT
        self._min_face_px = max(8, DEFAULT_MIN_FACE_PX)
        self._operator = DEFAULT_OPERATOR if DEFAULT_OPERATOR in ("gaussian", "mosaic") else "gaussian"
        self._max_throttle = max(1, DEFAULT_MAX_THROTTLE)

        self._load_models()
        self._load_haar_cascade()
        self._enforce_gpu_requirement()

    def gpu_status(self) -> Dict[str, object]:
        return {
            "blur_device": self._device,
            "cuda_available": self._cuda_available,
            "cuda_name": self._cuda_name,
            "ort_providers": list(self._ort_providers),
            "insightface_on_gpu": self._insightface_on_gpu,
            "insightface_loaded": self._insightface is not None,
            "require_gpu": _require_gpu(),
            "operator": self._operator,
        }

    def _enforce_gpu_requirement(self) -> None:
        if not _require_gpu():
            return
        problems: List[str] = []
        if not self._cuda_available or "cuda" not in self._device:
            problems.append("torch CUDA unavailable or BLUR_DEVICE is not cuda:*")
        if self._insightface is not None and not self._insightface_on_gpu:
            problems.append(
                "InsightFace has no CUDAExecutionProvider "
                "(Kaggle CUDA12: pip uninstall -y onnxruntime onnxruntime-gpu && "
                "pip install onnxruntime-gpu==1.19.2 — NOT 1.31/CUDA13 — then restart kernel; "
                "or %run install_kaggle_blur_gpu.py)"
            )
        if self._insightface is None and self._face_detect_mode != "heuristic":
            problems.append("InsightFace failed to load")
        if problems:
            msg = (
                "[blur] BLUR_REQUIRE_GPU: GPU blur not ready. "
                + "; ".join(problems)
                + ". Set notebook Accel=GPU T4 x2, then reinstall onnxruntime-gpu."
            )
            logger.error(msg)
            raise RuntimeError(msg)

    def _load_haar_cascade(self) -> None:
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            self._haar_cascade = None if cascade.empty() else cascade
        except Exception as exc:
            logger.warning("[blur] Haar cascade unavailable: %s", exc)
            self._haar_cascade = None

    def _load_models(self) -> None:
        from ultralytics import YOLO

        weights = os.getenv("BLUR_YOLO_WEIGHTS", "yolov8n.pt")
        if not Path(weights).exists() and not weights.endswith(".pt"):
            weights = "yolov8n.pt"
        logger.info(
            "[blur] Loading YOLO weights=%s device=%s cuda=%s (%s)",
            weights,
            self._device,
            self._cuda_available,
            self._cuda_name or "n/a",
        )
        self._yolo = YOLO(weights)

        face_weights = os.getenv("BLUR_FACE_WEIGHTS", "").strip()
        if face_weights and Path(face_weights).exists():
            self._face_yolo = YOLO(face_weights)
        else:
            self._face_yolo = self._yolo

        try:
            from insightface.app import FaceAnalysis
            import onnxruntime as ort

            # Pull CUDA/cuDNN libs from the torch / nvidia-* wheel tree when possible.
            try:
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
            except Exception as preload_exc:
                logger.warning("[blur] ort.preload_dlls failed: %s", preload_exc)

            available = list(ort.get_available_providers())
            self._ort_providers = available
            ort_ver = getattr(ort, "__version__", "?")
            logger.info(
                "[blur] onnxruntime %s providers=%s", ort_ver, available
            )

            providers: List[str]
            if (
                self._cuda_available
                and "cuda" in self._device
                and "CUDAExecutionProvider" in available
            ):
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self._insightface_on_gpu = True
            else:
                providers = ["CPUExecutionProvider"]
                self._insightface_on_gpu = False
                if self._cuda_available and "CUDAExecutionProvider" not in available:
                    logger.warning(
                        "[blur] CUDA torch OK but onnxruntime %s lacks CUDAExecutionProvider "
                        "(providers=%s). Kaggle/CUDA12 fix: pip uninstall -y onnxruntime "
                        "onnxruntime-gpu && pip install onnxruntime-gpu==1.19.2 "
                        "(or %%run install_kaggle_blur_gpu.py), then RESTART kernel. "
                        "onnxruntime-gpu>=1.27 needs CUDA 13 and will not work on Kaggle T4.",
                        ort_ver,
                        available,
                    )

            self._insightface = FaceAnalysis(
                name=os.getenv("BLUR_INSIGHTFACE_MODEL", "buffalo_l"),
                providers=providers,
            )
            use_cuda = self._insightface_on_gpu
            # Larger det_size helps small / profile faces in person crops
            det_w = int(os.getenv("BLUR_DET_SIZE", "640"))
            det_w = max(320, min(1024, det_w))
            self._insightface.prepare(
                ctx_id=0 if use_cuda else -1, det_size=(det_w, det_w)
            )
            logger.info(
                "[blur] InsightFace loaded ctx_id=%s on_gpu=%s providers=%s",
                0 if use_cuda else -1,
                self._insightface_on_gpu,
                providers,
            )
        except Exception as exc:
            logger.warning("[blur] InsightFace unavailable (%s)", exc)
            self._insightface = None
            self._insightface_on_gpu = False

    def _histogram_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        small = cv2.resize(face_bgr, (64, 64), interpolation=cv2.INTER_AREA)
        vec = small.astype(np.float32).reshape(-1)
        vec /= np.linalg.norm(vec) + 1e-9
        return vec

    def _insightface_embedding_from_image(
        self,
        image_bgr: np.ndarray,
        target_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        if self._insightface is None:
            return None
        try:
            faces = self._insightface.get(image_bgr)
        except Exception as exc:
            logger.warning("[blur] InsightFace.get failed: %s", exc)
            return None
        if not faces:
            return None
        if target_box is None:
            return self._embedding_from_face_obj(faces[0])
        best_face = max(
            faces,
            key=lambda f: _box_iou(tuple(map(int, f.bbox[:4])), target_box),
        )
        return self._embedding_from_face_obj(best_face)

    def _embedding_for_face_region(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        target = (x1, y1, x2, y2)

        if self._insightface is not None:
            pad = max(20, int(0.35 * max(x2 - x1, y2 - y1)))
            px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
            px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
            patch = frame[py1:py2, px1:px2]
            emb = self._insightface_embedding_from_image(patch, target)
            if emb is not None:
                return emb
            return self._insightface_embedding_from_image(frame, target)

        if self._face_detect_mode == "heuristic":
            crop = frame[y1:y2, x1:x2]
            return self._histogram_embedding(crop) if crop.size else None
        return None

    def _detect_face_boxes_insightface(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self._insightface is None:
            return []
        boxes: List[Tuple[int, int, int, int]] = []
        try:
            faces = self._insightface.get(image_bgr)
            for face in faces or []:
                bbox = getattr(face, "bbox", None)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = map(int, bbox[:4])
                if x2 - x1 >= 20 and y2 - y1 >= 20:
                    boxes.append((x1, y1, x2, y2))
        except Exception as exc:
            logger.warning("[blur] InsightFace detect failed: %s", exc)
        return boxes

    def _detect_face_boxes(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        boxes = self._detect_face_boxes_insightface(image_bgr)
        if boxes:
            return boxes

        boxes = []
        results = self._face_yolo.predict(
            image_bgr,
            verbose=False,
            device=self._device,
            classes=[0] if self._face_yolo is self._yolo else None,
        )
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue
                boxes.append((x1, y1, x2, y2))
        if boxes:
            return boxes

        if self._haar_cascade is not None:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            for (x, y, fw, fh) in self._haar_cascade.detectMultiScale(
                gray, 1.1, 4, minSize=(30, 30)
            ):
                boxes.append((x, y, x + fw, y + fh))
        return boxes

    def _embeddings_from_bgr(self, img: np.ndarray) -> List[np.ndarray]:
        """All face embeddings in an image (InsightFace or heuristic)."""
        face_embs: List[np.ndarray] = []
        if self._insightface is not None:
            try:
                faces = self._insightface.get(img)
                for face in faces or []:
                    emb = self._embedding_from_face_obj(face)
                    if emb is not None:
                        face_embs.append(emb)
            except Exception as exc:
                logger.warning("[blur] InsightFace embed failed: %s", exc)
        if not face_embs and self._face_detect_mode == "heuristic":
            for box in self._detect_face_boxes(img):
                emb = self._embedding_for_face_region(img, *box)
                if emb is not None:
                    face_embs.append(emb)
        return face_embs

    def build_master_signatures(self, reference_dir: Path) -> np.ndarray:
        """
        Gallery of reference embeddings for max-sim matching.

        - Each detected face in each reference image becomes a prototype
          (multi-angle side/front photos of the same person all help).
        - Horizontally flipped copies are added so mirrored profiles match.
        Returns array shape (N, D).
        """
        if self._face_detect_mode != "heuristic" and self._insightface is None:
            raise ValueError(
                "InsightFace is required for reference matching. "
                "Install insightface + onnxruntime-gpu, or set BLUR_FACE_DETECT=heuristic for ablation only."
            )

        masters: List[np.ndarray] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(p for p in reference_dir.iterdir() if p.suffix.lower() in exts)
        if not paths:
            raise ValueError("No reference images found in reference folder")

        use_flip = os.getenv("BLUR_REF_FLIP_AUG", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        ref_images_used = 0

        for path in paths:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("[blur] Could not read reference image: %s", path)
                continue

            face_embs = self._embeddings_from_bgr(img)
            flip_embs: List[np.ndarray] = []
            if use_flip and self._insightface is not None:
                flip_embs = self._embeddings_from_bgr(cv2.flip(img, 1))

            if not face_embs and not flip_embs:
                logger.warning("[blur] No face found in reference: %s", path.name)
                continue

            ref_images_used += 1
            # Centroid from real (non-flipped) faces + each pose + flip gallery
            if face_embs:
                mean_vec = np.mean(np.stack(face_embs, axis=0), axis=0)
                mean_vec = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
                masters.append(mean_vec.astype(np.float32))
                for emb in face_embs:
                    masters.append(emb.astype(np.float32))
            for emb in flip_embs:
                masters.append(emb.astype(np.float32))

            logger.info(
                "[blur] Ref %s → %d gallery vector(s) (faces=%d flip=%d)",
                path.name,
                (1 if face_embs else 0) + len(face_embs) + len(flip_embs),
                len(face_embs),
                len(flip_embs),
            )

        if not masters:
            raise ValueError(
                "No faces detected in reference images. "
                "Upload clear photos (front and/or side views) of the person(s) to blur."
            )

        dims = {m.shape[0] for m in masters}
        if len(dims) > 1:
            raise ValueError(
                f"Inconsistent embedding sizes {sorted(dims)} across reference images."
            )

        out = np.stack(masters, axis=0)
        logger.info(
            "[blur] Built gallery: %d embedding(s) from %d reference image(s) dim=%d",
            out.shape[0],
            ref_images_used,
            out.shape[1],
        )
        return out

    def _person_face_region(
        self, frame: np.ndarray, px1: int, py1: int, px2: int, py2: int
    ) -> Tuple[int, int, int, int]:
        """Upper-body head prior — wide enough for profiles / turned heads."""
        ph = py2 - py1
        pw = px2 - px1
        fy2 = py1 + int(ph * 0.50)
        # Narrow side crop used to kill profile faces; keep almost full person width
        fx1 = px1 + int(pw * 0.02)
        fx2 = px2 - int(pw * 0.02)
        sub = frame[py1:fy2, fx1:fx2]
        if sub.size == 0:
            return px1, py1, px2, py2
        if self._haar_cascade is not None:
            gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
            min_sz = max(16, min(24, self._min_face_px))
            for (x, y, fw, fh) in self._haar_cascade.detectMultiScale(
                gray, 1.1, 3, minSize=(min_sz, min_sz)
            ):
                return fx1 + x, py1 + y, fx1 + x + fw, py1 + y + fh
        return fx1, py1, fx2, fy2

    @staticmethod
    def _embedding_from_face_obj(face) -> Optional[np.ndarray]:
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
        if emb is None:
            return None
        vec = np.asarray(emb, dtype=np.float32).ravel()
        n = np.linalg.norm(vec)
        if n > 1e-9:
            vec = vec / n
        return vec

    def _faces_in_person_crop(
        self,
        frame: np.ndarray,
        px1: int,
        py1: int,
        px2: int,
        py2: int,
        *,
        pad_frac: float = 0.22,
        pad_x_extra: float = 0.12,
    ) -> List[FaceHit]:
        """SCRFD on a padded person crop; only faces that belong to this person."""
        if self._insightface is None:
            return []

        h, w = frame.shape[:2]
        pw, ph = max(1, px2 - px1), max(1, py2 - py1)
        pad_x = int((pad_frac + pad_x_extra) * pw)
        pad_y = int(pad_frac * ph)
        # Prefer head band crop (reduces stealing torso-neighbor faces)
        cy1 = max(0, py1 - pad_y)
        cy2 = min(h, py1 + int(0.70 * ph) + pad_y)
        cx1, cx2 = max(0, px1 - pad_x), min(w, px2 + pad_x)
        if cx2 <= cx1 or cy2 <= cy1:
            return []

        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return []

        try:
            faces = self._insightface.get(crop)
        except Exception as exc:
            logger.warning("[blur] InsightFace crop detect failed: %s", exc)
            return []

        person = (px1, py1, px2, py2)
        hits: List[FaceHit] = []
        min_px = max(16, self._min_face_px - 4)
        for face in faces or []:
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                continue
            fx1, fy1, fx2, fy2 = map(int, bbox[:4])
            fx1, fy1, fx2, fy2 = fx1 + cx1, fy1 + cy1, fx2 + cx1, fy2 + cy1
            fw, fh = fx2 - fx1, fy2 - fy1
            if fw < min_px or fh < min_px:
                continue
            assoc = _face_belongs_to_person((fx1, fy1, fx2, fy2), person)
            if assoc < 0:
                continue
            score = float(getattr(face, "det_score", 0.0) or 0.0)
            if score < 0.30:
                continue
            hits.append(
                FaceHit(
                    box=(fx1, fy1, fx2, fy2),
                    embedding=self._embedding_from_face_obj(face),
                    det_score=score * assoc,  # bake association into rank score
                    yaw=_face_yaw_deg(face),
                )
            )
        return hits

    def _best_face_for_person(
        self,
        frame: np.ndarray,
        px1: int,
        py1: int,
        px2: int,
        py2: int,
        *,
        prefer_continuity: bool = False,
        prev_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> FaceHit:
        use_insight = (
            self._face_detect_mode != "heuristic" and self._insightface is not None
        )
        if use_insight:
            hits = self._faces_in_person_crop(frame, px1, py1, px2, py2)
            if not hits:
                hits = self._faces_in_person_crop(
                    frame, px1, py1, px2, py2, pad_frac=0.32, pad_x_extra=0.22
                )
            if hits:
                # Prefer faces near previous lock box (temporal consistency)
                def _rank(hit: FaceHit) -> float:
                    area = max(1, (hit.box[2] - hit.box[0]) * (hit.box[3] - hit.box[1]))
                    score = hit.det_score * (area ** 0.5)
                    if prev_box is not None:
                        score += 2.0 * _box_iou(hit.box, prev_box)
                    return score

                return max(hits, key=_rank)

        # Locked continuity: reuse previous face box if available (never jump to torso)
        if prefer_continuity and prev_box is not None:
            return FaceHit(box=prev_box, embedding=None, det_score=0.2, yaw=45.0)

        box = self._person_face_region(frame, px1, py1, px2, py2)
        return FaceHit(
            box=box,
            embedding=None,
            det_score=0.0,
            yaw=0.0,
        )

    def process_video(
        self,
        video_path: Path,
        output_path: Path,
        masters: np.ndarray,
        *,
        match_threshold: float = DEFAULT_MATCH_THRESHOLD,
        throttle: int = 3,
        grace: int = 30,
        min_match_streak: int = 2,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, object]:
        if masters.ndim == 1:
            masters = masters.reshape(1, -1)
        if masters.ndim != 2 or masters.shape[0] < 1:
            raise ValueError("masters must be shape (N, D) with N>=1")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        start_frame = int((start_sec or 0) * fps)
        end_frame = int(end_sec * fps) if end_sec is not None and end_sec > 0 else total_frames
        if end_frame <= 0:
            end_frame = total_frames

        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = output_path.with_name(output_path.stem + "_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {raw_path}")

        active_tracks: Dict[int, int] = {}
        match_streak: Dict[int, int] = {}
        locked_targets: Set[int] = set()
        tracks_ever_locked: Set[int] = set()
        lock_identity: Dict[int, int] = {}
        pending_reattach: List[Dict[str, object]] = []
        # Ghost hold: keep blurring after brief track loss (stops on/off flicker)
        ghost_boxes: Dict[int, Tuple[Tuple[int, int, int, int], float, int]] = {}
        # tid -> last yaw for pad consistency
        last_yaw: Dict[int, float] = {}

        frame_idx = 0
        processed = 0
        throttle = max(1, int(throttle))
        grace = max(1, int(grace))
        # Slightly longer streak reduces accidental locks on bystanders
        min_streak = max(2, int(min_match_streak))
        master_dim = int(masters.shape[1])
        num_identities = int(masters.shape[0])
        duration_est = total_frames / max(fps, 1.0)
        box_alpha = float(os.getenv("BLUR_BOX_SMOOTH", "0.30"))
        ghost_ttl = max(grace, int(os.getenv("BLUR_GHOST_FRAMES", str(max(grace, 45)))))
        min_det_for_lock = float(os.getenv("BLUR_MIN_DET_SCORE", "0.40"))

        if duration_est > 300:
            throttle = max(throttle, min(self._max_throttle, 4))
        elif duration_est > 120:
            throttle = max(throttle, min(self._max_throttle, 3))
        elif duration_est > 60:
            throttle = max(throttle, min(self._max_throttle, 3))
        throttle = min(throttle, self._max_throttle)

        infer_max = int(os.getenv("BLUR_INFER_MAX_DIM", "960"))
        last_face_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        last_embeddings: Dict[int, np.ndarray] = {}
        frames_to_process = max(1, end_frame - start_frame)

        logger.info(
            "[blur] identities=%d match_threshold=%.2f min_streak=%d throttle=%d "
            "device=%s face_detect=%s operator=%s on_gpu_if=%s ~%.0fs video",
            num_identities,
            match_threshold,
            min_streak,
            throttle,
            self._device,
            self._face_detect_mode if self._insightface is not None else "heuristic",
            self._operator,
            self._insightface_on_gpu,
            duration_est,
        )

        def _report_progress() -> None:
            if progress_cb and frames_to_process > 0:
                pct = min(99.0, (processed / frames_to_process) * 100.0)
                progress_cb(pct, "Processing frames")

        def _blur_box(
            frame: np.ndarray,
            box: Tuple[int, int, int, int],
            *,
            yaw: float = 0.0,
        ) -> None:
            fx1, fy1, fx2, fy2 = box
            if abs(yaw) >= 25:
                bw, bh = max(1, fx2 - fx1), max(1, fy2 - fy1)
                pad = int(0.10 * max(bw, bh))
                fx1, fy1 = fx1 - pad, fy1 - pad
                fx2, fy2 = fx2 + pad, fy2 + pad
            _apply_face_anonymize(
                frame, fx1, fy1, fx2, fy2, operator=self._operator
            )

        def _apply_all_locks(frame: np.ndarray) -> None:
            """Blur every locked / ghost box every frame (no temporal holes)."""
            for tid in locked_targets:
                box = last_face_boxes.get(tid)
                if box:
                    _blur_box(frame, box, yaw=last_yaw.get(tid, 0.0))
            for tid, (box, yaw, _ttl) in list(ghost_boxes.items()):
                if tid not in locked_targets:
                    _blur_box(frame, box, yaw=yaw)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            in_range = start_frame <= frame_idx < end_frame
            if in_range:
                # Update boxes every frame once locked (smooth tracking); else throttle re-ID
                detect_interval = 1 if locked_targets else throttle
                run_detect = frame_idx % detect_interval == 0

                if run_detect:
                    infer_frame = frame
                    scale = 1.0
                    fh, fw = frame.shape[:2]
                    if max(fh, fw) > infer_max:
                        scale = infer_max / max(fh, fw)
                        infer_frame = cv2.resize(
                            frame,
                            (int(fw * scale), int(fh * scale)),
                            interpolation=cv2.INTER_AREA,
                        )

                    results = self._yolo.track(
                        infer_frame,
                        persist=True,
                        verbose=False,
                        device=self._device,
                        classes=[0],
                        tracker=os.getenv("BLUR_TRACKER", "botsort.yaml"),
                        imgsz=640,
                    )
                    current_ids: Set[int] = set()
                    frame_matches: List[Tuple[int, float, int, Optional[np.ndarray]]] = []
                    track_face: Dict[int, FaceHit] = {}

                    for r in results:
                        if r.boxes is None or r.boxes.id is None:
                            continue
                        ids = r.boxes.id.int().cpu().tolist()
                        xyxy = r.boxes.xyxy.cpu().numpy()
                        for tid, box in zip(ids, xyxy):
                            current_ids.add(tid)
                            bx1, by1, bx2, by2 = box[:4]
                            if scale != 1.0:
                                bx1, by1, bx2, by2 = (
                                    bx1 / scale,
                                    by1 / scale,
                                    bx2 / scale,
                                    by2 / scale,
                                )
                            px1, py1, px2, py2 = map(int, (bx1, by1, bx2, by2))
                            is_locked = tid in locked_targets
                            prev_box = last_face_boxes.get(tid)
                            face_hit = self._best_face_for_person(
                                frame,
                                px1,
                                py1,
                                px2,
                                py2,
                                prefer_continuity=is_locked,
                                prev_box=prev_box,
                            )
                            # EMA smooth — prevents glitchy jump between frames
                            smooth = _smooth_box(prev_box, face_hit.box, alpha=box_alpha)
                            face_hit = FaceHit(
                                box=smooth,
                                embedding=face_hit.embedding,
                                det_score=face_hit.det_score,
                                yaw=face_hit.yaw,
                            )
                            track_face[tid] = face_hit
                            last_face_boxes[tid] = smooth
                            last_yaw[tid] = face_hit.yaw

                            if is_locked:
                                continue

                            # Re-ID only on throttle ticks
                            if frame_idx % throttle != 0:
                                continue

                            # ONLY lock from real SCRFD+ArcFace face embeddings
                            # (torso/heuristic crops caused false positives on bystanders)
                            emb = face_hit.embedding
                            if (
                                emb is None
                                or emb.shape[0] != master_dim
                                or face_hit.det_score < min_det_for_lock
                            ):
                                match_streak[tid] = 0
                                continue

                            last_embeddings[tid] = emb
                            sim, mid = _max_sim_to_masters(emb, masters)
                            thr = _effective_match_threshold(
                                match_threshold, face_hit.yaw
                            )
                            if sim >= thr:
                                frame_matches.append((tid, sim, mid, emb))
                            else:
                                match_streak[tid] = 0

                    # Re-attach after ID switch
                    if frame_idx % throttle == 0 and pending_reattach:
                        still_pending: List[Dict[str, object]] = []
                        for pend in pending_reattach:
                            pend["miss"] = int(pend["miss"]) + max(1, throttle)
                            if int(pend["miss"]) > grace:
                                continue
                            claimed = False
                            for tid, sim, mid, emb in frame_matches:
                                if tid in locked_targets:
                                    continue
                                if mid != int(pend["master_idx"]):
                                    continue
                                last_box = pend["box"]  # type: ignore[assignment]
                                new_box = track_face[tid].box
                                iou = _box_iou(new_box, last_box)  # type: ignore[arg-type]
                                max_side = max(
                                    last_box[2] - last_box[0],  # type: ignore[index]
                                    last_box[3] - last_box[1],  # type: ignore[index]
                                    1,
                                )
                                near = _box_center_dist(new_box, last_box) <= 1.2 * max_side  # type: ignore[arg-type]
                                thr = _effective_match_threshold(
                                    match_threshold, track_face[tid].yaw
                                )
                                if sim >= thr and (iou >= 0.12 or near):
                                    locked_targets.add(tid)
                                    tracks_ever_locked.add(tid)
                                    lock_identity[tid] = mid
                                    active_tracks[tid] = 0
                                    match_streak[tid] = min_streak
                                    ghost_boxes.pop(tid, None)
                                    claimed = True
                                    break
                            if not claimed:
                                still_pending.append(pend)
                        pending_reattach = still_pending

                    if frame_idx % throttle == 0:
                        matched_ids = {t for t, _, _, _ in frame_matches}
                        for tid, sim, mid, emb in frame_matches:
                            if tid in locked_targets:
                                continue
                            match_streak[tid] = match_streak.get(tid, 0) + 1
                            if match_streak[tid] >= min_streak:
                                locked_targets.add(tid)
                                tracks_ever_locked.add(tid)
                                lock_identity[tid] = mid
                                active_tracks[tid] = 0
                                ghost_boxes.pop(tid, None)
                                logger.info(
                                    "[blur] Locked track %s sim=%.3f master=%s",
                                    tid,
                                    sim,
                                    mid,
                                )
                        for tid in list(match_streak.keys()):
                            if tid not in matched_ids and tid not in locked_targets:
                                match_streak[tid] = 0

                    for tid in list(locked_targets):
                        if tid not in current_ids:
                            active_tracks[tid] = active_tracks.get(tid, 0) + detect_interval
                            box = last_face_boxes.get(tid)
                            if box is not None:
                                ghost_boxes[tid] = (
                                    box,
                                    last_yaw.get(tid, 0.0),
                                    ghost_ttl,
                                )
                            if active_tracks[tid] > grace:
                                mid = lock_identity.get(tid, 0)
                                emb = last_embeddings.get(tid)
                                if box is not None:
                                    pending_reattach.append(
                                        {
                                            "master_idx": mid,
                                            "box": box,
                                            "emb": emb,
                                            "miss": 0,
                                        }
                                    )
                                locked_targets.discard(tid)
                                active_tracks.pop(tid, None)
                                match_streak.pop(tid, None)
                                lock_identity.pop(tid, None)
                                # keep last_face_boxes for ghost briefly
                        else:
                            active_tracks[tid] = 0
                            ghost_boxes.pop(tid, None)

                # Decay ghosts
                for tid in list(ghost_boxes.keys()):
                    box, yaw, ttl = ghost_boxes[tid]
                    ttl -= 1
                    if ttl <= 0 or tid in locked_targets:
                        ghost_boxes.pop(tid, None)
                    else:
                        ghost_boxes[tid] = (box, yaw, ttl)

                # Always paint locks this frame (eliminates inter-detect flicker)
                _apply_all_locks(frame)

            writer.write(frame)
            frame_idx += 1
            if in_range:
                processed += 1
                if processed % 30 == 0:
                    _report_progress()

        cap.release()
        writer.release()

        if progress_cb:
            progress_cb(99.0, "Encoding output")

        # OpenCV writes every frame (blur only in [start,end]) → 1:1 with source audio
        if _transcode_for_browser(
            raw_path,
            output_path,
            audio_src=video_path,
        ):
            try:
                raw_path.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            logger.warning(
                "[blur] FFmpeg transcode failed; using raw mp4v (may not play in browser, no audio)"
            )
            if raw_path.exists() and not output_path.exists():
                raw_path.replace(output_path)

        duration_sec = processed / fps if fps > 0 else 0.0
        return {
            "target_ids_blurred": len(locked_targets),
            "tracks_ever_locked": len(tracks_ever_locked),
            "num_identities": num_identities,
            "total_frames": processed,
            "fps": fps,
            "width": width,
            "height": height,
            "video_duration_sec": round(duration_sec, 2),
            "match_threshold": match_threshold,
            "device": self._device,
            "insightface_on_gpu": self._insightface_on_gpu,
            "operator": self._operator,
            "browser_ready": output_path.exists(),
        }


def get_blur_engine() -> BlurEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = BlurEngine()
    return _ENGINE


def blur_gpu_status() -> Dict[str, object]:
    """GPU / ORT status without forcing full engine load if already loaded."""
    if _ENGINE is not None:
        return _ENGINE.gpu_status()
    # Lightweight probe without loading YOLO/InsightFace weights
    status: Dict[str, object] = {
        "blur_device": os.getenv("BLUR_DEVICE", ""),
        "cuda_available": False,
        "cuda_name": "",
        "ort_providers": [],
        "insightface_on_gpu": False,
        "insightface_loaded": False,
        "require_gpu": _require_gpu(),
        "engine_loaded": False,
    }
    try:
        import torch

        status["cuda_available"] = bool(torch.cuda.is_available())
        if status["cuda_available"]:
            status["cuda_name"] = torch.cuda.get_device_name(0)
            if not status["blur_device"]:
                status["blur_device"] = "cuda:0"
    except Exception:
        pass
    try:
        import onnxruntime as ort

        try:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        except Exception:
            pass
        status["ort_version"] = getattr(ort, "__version__", "")
        status["ort_providers"] = list(ort.get_available_providers())
        status["insightface_on_gpu"] = (
            status["cuda_available"]
            and "CUDAExecutionProvider" in status["ort_providers"]
        )
    except Exception:
        pass
    return status


def anonymize_video(
    video_path: Path,
    reference_dir: Path,
    output_path: Path,
    *,
    match_threshold: float = DEFAULT_MATCH_THRESHOLD,
    throttle: int = 3,
    grace: int = 30,
    min_match_streak: int = 2,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    progress_cb: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, object]:
    """Blur faces in video that match person(s) in reference images."""
    t0 = time.perf_counter()
    with _PROCESS_LOCK:
        engine = get_blur_engine()
        masters = engine.build_master_signatures(reference_dir)
        stats = engine.process_video(
            video_path,
            output_path,
            masters,
            match_threshold=match_threshold,
            throttle=throttle,
            grace=grace,
            min_match_streak=min_match_streak,
            start_sec=start_sec,
            end_sec=end_sec,
            progress_cb=progress_cb,
        )
    elapsed = time.perf_counter() - t0
    stats["processing_time_sec"] = round(elapsed, 2)
    stats["output_path"] = str(output_path)
    return stats


def blur_model_loaded() -> bool:
    return _ENGINE is not None


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="CLI test for blur_service")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--refs", type=Path, required=True, help="Directory of reference images")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    args = parser.parse_args()
    out = anonymize_video(
        args.video,
        args.refs,
        args.output,
        match_threshold=args.threshold,
    )
    print(out)
