"""
Reference-image face anonymization for NeuroClip.

Pipeline:
  1. Build master face embedding(s) from reference images (YOLO person/face + optional InsightFace).
  2. Process video frames with YOLO tracking; match detected faces to master signatures.
  3. Apply Gaussian blur to matched regions; propagate via track IDs with grace frames.

Configure weights via BLUR_YOLO_WEIGHTS (default: yolov8n.pt auto-download).
Optional: BLUR_FACE_WEIGHTS for a dedicated face detector .pt file.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROCESS_LOCK = threading.Lock()
_ENGINE: Optional["BlurEngine"] = None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    if a.shape[0] != b.shape[0]:
        logger.warning(
            "[blur] embedding dim mismatch: %d vs %d — skipping compare",
            a.shape[0],
            b.shape[0],
        )
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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


def _gaussian_blur_roi(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, ksize: int = 51) -> None:
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    k = ksize if ksize % 2 == 1 else ksize + 1
    k = max(15, min(k, min(x2 - x1, y2 - y1) | 1))
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


class BlurEngine:
    """Lazy-loaded YOLO + optional InsightFace for reference-based face blurring."""

    def __init__(self) -> None:
        self._device = os.getenv("BLUR_DEVICE", "").strip()
        if not self._device:
            try:
                import torch

                self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"

        self._yolo = None
        self._face_yolo = None
        self._insightface = None
        self._load_models()

    def _load_models(self) -> None:
        from ultralytics import YOLO

        weights = os.getenv("BLUR_YOLO_WEIGHTS", "yolov8n.pt")
        if not Path(weights).exists() and not weights.endswith(".pt"):
            weights = "yolov8n.pt"
        logger.info("[blur] Loading YOLO weights: %s on %s", weights, self._device)
        self._yolo = YOLO(weights)

        face_weights = os.getenv("BLUR_FACE_WEIGHTS", "").strip()
        if face_weights and Path(face_weights).exists():
            self._face_yolo = YOLO(face_weights)
        else:
            self._face_yolo = self._yolo

        try:
            from insightface.app import FaceAnalysis

            # Prefer CPU provider when CUDA onnxruntime is not installed (common on Kaggle)
            providers = ["CPUExecutionProvider"]
            try:
                import onnxruntime as ort

                available = set(ort.get_available_providers())
                if "CUDAExecutionProvider" in available and "cuda" in self._device:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:
                pass
            self._insightface = FaceAnalysis(
                name=os.getenv("BLUR_INSIGHTFACE_MODEL", "buffalo_l"),
                providers=providers,
            )
            self._insightface.prepare(ctx_id=0 if "cuda" in self._device else -1, det_size=(640, 640))
            logger.info("[blur] InsightFace loaded")
        except Exception as exc:
            logger.warning("[blur] InsightFace unavailable (%s); using histogram fallback", exc)
            self._insightface = None

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
            return np.asarray(faces[0].embedding, dtype=np.float32)
        best_face = max(
            faces,
            key=lambda f: _box_iou(
                tuple(map(int, f.bbox[:4])),
                target_box,
            ),
        )
        emb = getattr(best_face, "embedding", None)
        return np.asarray(emb, dtype=np.float32) if emb is not None else None

    def _embedding_for_face_region(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> Optional[np.ndarray]:
        """512-d InsightFace when available; histogram fallback only if InsightFace disabled."""
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
            emb = self._insightface_embedding_from_image(frame, target)
            if emb is not None:
                return emb
            return None

        crop = frame[y1:y2, x1:x2]
        return self._histogram_embedding(crop) if crop.size else None

    def _detect_face_boxes_insightface(self, image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """InsightFace detector — best for reference portrait photos."""
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
        """Return face/person boxes as (x1,y1,x2,y2)."""
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
                w, h = x2 - x1, y2 - y1
                if w < 20 or h < 20:
                    continue
                boxes.append((x1, y1, x2, y2))
        if boxes:
            return boxes

        # Haar fallback for reference stills
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        for (x, y, fw, fh) in cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30)):
            boxes.append((x, y, x + fw, y + fh))
        return boxes

    def build_master_signatures(self, reference_dir: Path) -> np.ndarray:
        """Average embeddings from all faces found in reference images."""
        embeddings: List[np.ndarray] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(p for p in reference_dir.iterdir() if p.suffix.lower() in exts)
        if not paths:
            raise ValueError("No reference images found in reference folder")

        for path in paths:
            img = cv2.imread(str(path))
            if img is None:
                logger.warning("[blur] Could not read reference image: %s", path)
                continue

            added_for_image = False
            if self._insightface is not None:
                try:
                    faces = self._insightface.get(img)
                    for face in faces or []:
                        emb = getattr(face, "embedding", None)
                        if emb is not None:
                            embeddings.append(np.asarray(emb, dtype=np.float32))
                            added_for_image = True
                except Exception as exc:
                    logger.warning("[blur] InsightFace embed on %s failed: %s", path.name, exc)

            if not added_for_image:
                for box in self._detect_face_boxes(img):
                    emb = self._embedding_for_face_region(img, *box)
                    if emb is not None:
                        embeddings.append(emb)
                        added_for_image = True

            if not added_for_image:
                logger.warning("[blur] No face found in reference: %s", path.name)

        if not embeddings:
            raise ValueError(
                "No faces detected in reference images. "
                "Upload clear front-facing photos of the person(s) to blur."
            )

        dims = {e.shape[0] for e in embeddings}
        if len(dims) > 1:
            raise ValueError(
                f"Inconsistent embedding sizes {sorted(dims)} across reference images. "
                "Use similar photo types or ensure InsightFace detects all references."
            )
        master = np.mean(np.stack(embeddings, axis=0), axis=0)
        master /= np.linalg.norm(master) + 1e-9
        logger.info(
            "[blur] Master signature dim=%d from %d face(s) across %d file(s)",
            master.shape[0],
            len(embeddings),
            len(paths),
        )
        return master.astype(np.float32)

    def _person_face_region(
        self, frame: np.ndarray, px1: int, py1: int, px2: int, py2: int
    ) -> Tuple[int, int, int, int]:
        """Estimate face region inside a person bounding box (upper ~40%)."""
        ph = py2 - py1
        pw = px2 - px1
        fy2 = py1 + int(ph * 0.45)
        fx1 = px1 + int(pw * 0.15)
        fx2 = px2 - int(pw * 0.15)
        sub = frame[py1:fy2, fx1:fx2]
        if sub.size == 0:
            return px1, py1, px2, py2
        for (x, y, fw, fh) in cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        ).detectMultiScale(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), 1.1, 4, minSize=(24, 24)):
            return fx1 + x, py1 + y, fx1 + x + fw, py1 + y + fh
        return fx1, py1, fx2, fy2

    def process_video(
        self,
        video_path: Path,
        output_path: Path,
        master: np.ndarray,
        *,
        match_threshold: float = 0.65,
        throttle: int = 3,
        grace: int = 30,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> Dict[str, object]:
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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {output_path}")

        active_tracks: Dict[int, int] = {}  # track_id -> frames since last match
        locked_targets: Set[int] = set()
        frame_idx = 0
        processed = 0
        throttle = max(1, int(throttle))
        grace = max(1, int(grace))
        master_dim = int(master.shape[0])
        logger.info("[blur] Video match using master_dim=%d insightface=%s", master_dim, self._insightface is not None)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            in_range = start_frame <= frame_idx < end_frame
            if in_range and frame_idx % throttle == 0:
                results = self._yolo.track(
                    frame,
                    persist=True,
                    verbose=False,
                    device=self._device,
                    classes=[0],
                    tracker=os.getenv("BLUR_TRACKER", "botsort.yaml"),
                )
                current_ids: Set[int] = set()
                for r in results:
                    if r.boxes is None or r.boxes.id is None:
                        continue
                    ids = r.boxes.id.int().cpu().tolist()
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    for tid, box in zip(ids, xyxy):
                        current_ids.add(tid)
                        px1, py1, px2, py2 = map(int, box[:4])
                        fx1, fy1, fx2, fy2 = self._person_face_region(frame, px1, py1, px2, py2)
                        emb = self._embedding_for_face_region(frame, fx1, fy1, fx2, fy2)
                        if emb is None or emb.shape[0] != master_dim:
                            continue
                        sim = _cosine_similarity(master, emb)
                        if sim >= match_threshold:
                            locked_targets.add(tid)
                            active_tracks[tid] = 0
                        elif tid in locked_targets:
                            active_tracks[tid] = active_tracks.get(tid, 0) + 1
                            if active_tracks[tid] > grace:
                                locked_targets.discard(tid)
                                active_tracks.pop(tid, None)

                # Decay tracks not seen this detection frame
                for tid in list(locked_targets):
                    if tid not in current_ids:
                        active_tracks[tid] = active_tracks.get(tid, 0) + throttle
                        if active_tracks[tid] > grace:
                            locked_targets.discard(tid)
                            active_tracks.pop(tid, None)

            if in_range and locked_targets:
                results = self._yolo.track(
                    frame,
                    persist=True,
                    verbose=False,
                    device=self._device,
                    classes=[0],
                    tracker=os.getenv("BLUR_TRACKER", "botsort.yaml"),
                )
                for r in results:
                    if r.boxes is None or r.boxes.id is None:
                        continue
                    ids = r.boxes.id.int().cpu().tolist()
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    for tid, box in zip(ids, xyxy):
                        if tid not in locked_targets:
                            continue
                        px1, py1, px2, py2 = map(int, box[:4])
                        fx1, fy1, fx2, fy2 = self._person_face_region(frame, px1, py1, px2, py2)
                        pad = int(0.08 * max(fx2 - fx1, fy2 - fy1))
                        _gaussian_blur_roi(
                            frame,
                            fx1 - pad,
                            fy1 - pad,
                            fx2 + pad,
                            fy2 + pad,
                        )

            writer.write(frame)
            frame_idx += 1
            if in_range:
                processed += 1

        cap.release()
        writer.release()

        return {
            "target_ids_blurred": len(locked_targets),
            "total_frames": processed,
            "fps": fps,
            "width": width,
            "height": height,
        }


def get_blur_engine() -> BlurEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = BlurEngine()
    return _ENGINE


def anonymize_video(
    video_path: Path,
    reference_dir: Path,
    output_path: Path,
    *,
    match_threshold: float = 0.65,
    throttle: int = 3,
    grace: int = 30,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Dict[str, object]:
    """
    Blur faces in video that match person(s) in reference images.

    Returns dict with target_ids_blurred, total_frames, processing_time_sec, etc.
    """
    t0 = time.perf_counter()
    with _PROCESS_LOCK:
        engine = get_blur_engine()
        master = engine.build_master_signatures(reference_dir)
        stats = engine.process_video(
            video_path,
            output_path,
            master,
            match_threshold=match_threshold,
            throttle=throttle,
            grace=grace,
            start_sec=start_sec,
            end_sec=end_sec,
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
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    out = anonymize_video(
        args.video,
        args.refs,
        args.output,
        match_threshold=args.threshold,
    )
    print(out)
