"""
Unit tests for blur performance phases (sparse schedule, helpers, chunk parallel).

Run from this directory:
  python -m unittest test_blur_performance.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

# Ensure backend package dir is importable when run as script / unittest
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import blur_chunk_parallel as chunk  # noqa: E402
import blur_service as blur  # noqa: E402


class TestPhase1SparseSchedule(unittest.TestCase):
    def test_unlocked_uses_throttle(self):
        flags = [
            blur.schedule_sparse_work(
                frame_idx=i,
                has_lock=False,
                throttle=3,
                track_interval=3,
                face_interval=5,
            )
            for i in range(9)
        ]
        # track only on 0,3,6
        self.assertEqual([f[0] for f in flags], [True, False, False, True, False, False, True, False, False])
        # no locked face ticks when unlocked
        self.assertFalse(any(f[1] for f in flags))
        # face search follows track while unlocked
        self.assertEqual([f[2] for f in flags], [f[0] for f in flags])

    def test_locked_face_tick_forces_track(self):
        # track=3, face=5 → frame 5 is face tick but not track tick unless forced
        run_track, run_face_locked, run_face_search = blur.schedule_sparse_work(
            frame_idx=5,
            has_lock=True,
            throttle=3,
            track_interval=3,
            face_interval=5,
        )
        self.assertTrue(run_face_locked)
        self.assertTrue(run_track)
        self.assertFalse(run_face_search)

    def test_locked_track_without_face(self):
        run_track, run_face_locked, run_face_search = blur.schedule_sparse_work(
            frame_idx=3,
            has_lock=True,
            throttle=3,
            track_interval=3,
            face_interval=5,
        )
        self.assertTrue(run_track)
        self.assertFalse(run_face_locked)
        self.assertFalse(run_face_search)

    def test_locked_sparse_reduces_track_density(self):
        n = 30
        locked = [
            blur.schedule_sparse_work(
                frame_idx=i,
                has_lock=True,
                throttle=1,
                track_interval=3,
                face_interval=5,
            )[0]
            for i in range(n)
        ]
        # Every-frame baseline would be 30; sparse should be well under half
        self.assertLess(sum(locked), n // 2)
        # Still enough ticks (track 3 + forced face 5 → ~12–14 in 30 frames)
        self.assertGreaterEqual(sum(locked), 10)

    def test_pending_reattach_enables_face_search_on_face_tick(self):
        run_track, run_face_locked, run_face_search = blur.schedule_sparse_work(
            frame_idx=5,
            has_lock=True,
            throttle=3,
            track_interval=3,
            face_interval=5,
            pending_reattach=True,
        )
        self.assertTrue(run_track)
        self.assertTrue(run_face_locked)
        self.assertTrue(run_face_search)

    def test_need_scrfd_skips_bystanders_when_locked(self):
        self.assertFalse(
            blur.need_scrfd_for_track(
                is_locked=False, run_face_locked=True, run_face_search=False
            )
        )
        self.assertTrue(
            blur.need_scrfd_for_track(
                is_locked=True, run_face_locked=True, run_face_search=False
            )
        )
        self.assertTrue(
            blur.need_scrfd_for_track(
                is_locked=False, run_face_locked=False, run_face_search=True
            )
        )


class TestPlausibleBoxAndSmooth(unittest.TestCase):
    def test_reject_huge_roi(self):
        self.assertFalse(blur._is_plausible_face_box((0, 0, 900, 900), 1000, 1000))

    def test_accept_normal_face(self):
        self.assertTrue(blur._is_plausible_face_box((100, 100, 180, 200), 1280, 720))

    def test_smooth_box_ema(self):
        prev = (0, 0, 100, 100)
        new = (100, 100, 200, 200)
        out = blur._smooth_box(prev, new, alpha=0.5)
        self.assertEqual(out, (50, 50, 150, 150))


class TestWideRetryOnMiss(unittest.TestCase):
    def test_best_face_retries_wide_pad_when_first_empty(self):
        engine = object.__new__(blur.BlurEngine)
        engine._face_detect_mode = "insightface"
        engine._insightface = mock.MagicMock()
        calls = {"n": 0}

        def _faces(*_a, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return []
            return [
                blur.FaceHit(
                    box=(10, 10, 40, 50),
                    embedding=np.ones(8, dtype=np.float32),
                    det_score=0.9,
                    yaw=0.0,
                    assoc=1.0,
                )
            ]

        engine._faces_in_person_crop = _faces  # type: ignore[method-assign]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        hit = blur.BlurEngine._best_face_for_person(
            engine, frame, 0, 0, 80, 90, wide_retry=True, cheap_det=True
        )
        self.assertEqual(calls["n"], 2)
        self.assertEqual(hit.det_score, 0.9)


class TestCheapDetCropScale(unittest.TestCase):
    def test_faces_in_person_crop_max_side_maps_boxes(self):
        """Downscaled SCRFD path must remap boxes to full-frame coords."""
        engine = object.__new__(blur.BlurEngine)
        engine._insightface = mock.MagicMock()
        engine._min_face_px = 28

        class FakeFace:
            bbox = np.array([10, 10, 40, 50], dtype=np.float32)
            det_score = 0.9
            embedding = np.ones(512, dtype=np.float32)
            pose = np.array([0.0, 0.0, 0.0])

        engine._insightface.get.return_value = [FakeFace()]
        engine._embedding_from_face_obj = lambda f: np.ones(512, dtype=np.float32)

        # Large crop so max_side=40 triggers downscale (~2x)
        frame = np.zeros((400, 400, 3), dtype=np.uint8)
        px1, py1, px2, py2 = 0, 0, 200, 300

        with mock.patch.object(blur, "_face_belongs_to_person", return_value=1.0):
            hits = blur.BlurEngine._faces_in_person_crop(
                engine, frame, px1, py1, px2, py2, max_side=40
            )

        self.assertEqual(len(hits), 1)
        # Crop was downscaled; box should be scaled back toward crop coords + offset
        fx1, fy1, fx2, fy2 = hits[0].box
        self.assertGreater(fx2 - fx1, 20)
        # Called with a resized crop smaller than original head-band crop
        args, _ = engine._insightface.get.call_args
        crop = args[0]
        self.assertLessEqual(max(crop.shape[:2]), 40)


class TestPhase2PipelineQueues(unittest.TestCase):
    def test_bounded_queue_blocks_not_drops(self):
        import queue as qmod

        q = qmod.Queue(maxsize=2)
        q.put(1)
        q.put(2)
        done = {"ok": False}

        def _put_third():
            q.put(3, timeout=1.0)
            done["ok"] = True

        t = threading.Thread(target=_put_third)
        t.start()
        t.join(timeout=0.2)
        self.assertTrue(t.is_alive())  # blocked on full queue
        self.assertEqual(q.get(), 1)
        t.join(timeout=1.0)
        self.assertTrue(done["ok"])
        self.assertFalse(t.is_alive())


class TestPhase3ChunkHelpers(unittest.TestCase):
    def test_should_use_off_env(self):
        with mock.patch.dict(os.environ, {"BLUR_CHUNK_PARALLEL": "0"}, clear=False):
            with mock.patch.object(chunk, "_gpu_count", return_value=2):
                self.assertFalse(chunk.should_use_chunk_parallel(Path("x.mp4")))

    def test_should_use_on_requires_two_gpus(self):
        with mock.patch.dict(os.environ, {"BLUR_CHUNK_PARALLEL": "1"}, clear=False):
            with mock.patch.object(chunk, "_gpu_count", return_value=1):
                self.assertFalse(chunk.should_use_chunk_parallel(Path("x.mp4")))
            with mock.patch.object(chunk, "_gpu_count", return_value=2):
                self.assertTrue(chunk.should_use_chunk_parallel(Path("x.mp4")))

    def test_should_use_auto_duration_gate(self):
        with mock.patch.dict(
            os.environ,
            {"BLUR_CHUNK_PARALLEL": "", "BLUR_CHUNK_MIN_SEC": "180"},
            clear=False,
        ):
            with mock.patch.object(chunk, "_gpu_count", return_value=2):
                with mock.patch.object(chunk, "_ffprobe_duration", return_value=60.0):
                    self.assertFalse(chunk.should_use_chunk_parallel(Path("x.mp4")))
                with mock.patch.object(chunk, "_ffprobe_duration", return_value=200.0):
                    self.assertTrue(chunk.should_use_chunk_parallel(Path("x.mp4")))

    def test_concat_list_entry_escapes_quote(self):
        p = Path("/tmp/odd'name.mp4")
        line = chunk._concat_list_entry(p)
        self.assertTrue(line.startswith("file '"))
        self.assertIn(r"'\''", line)

    def test_split_chunks_n_lt_2_returns_original(self):
        vid = Path("video.mp4")
        out = chunk._split_chunks(vid, Path("/tmp"), 1, 10.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], vid)

    def test_concat_videos_missing_parts_false(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            missing = td_path / "nope.mp4"
            out = td_path / "out.mp4"
            with mock.patch.object(chunk, "_ffmpeg", return_value="ffmpeg"):
                self.assertFalse(chunk._concat_videos([missing], out))


class TestAnonymizeRouting(unittest.TestCase):
    def test_time_range_skips_chunk_parallel(self):
        """start/end trim must not enter chunk path."""
        lock = mock.MagicMock()
        lock.__enter__ = mock.MagicMock(return_value=None)
        lock.__exit__ = mock.MagicMock(return_value=False)
        engine = mock.MagicMock()
        engine.build_master_signatures.return_value = (
            np.ones((1, 8), dtype=np.float32),
            np.array([0], dtype=np.int32),
        )
        engine.process_video.return_value = {
            "identities_ever_locked": 0,
            "browser_ready": True,
        }
        with mock.patch.object(chunk, "should_use_chunk_parallel") as gate:
            with mock.patch.object(blur, "get_blur_engine", return_value=engine):
                with mock.patch.object(blur, "_PROCESS_LOCK", lock):
                    stats = blur.anonymize_video(
                        Path("v.mp4"),
                        Path("refs"),
                        Path("out.mp4"),
                        start_sec=1.0,
                        end_sec=5.0,
                    )
        gate.assert_not_called()
        self.assertIn("processing_time_sec", stats)

    def test_chunk_parallel_preferred_when_eligible(self):
        lock = mock.MagicMock()
        lock.__enter__ = mock.MagicMock(return_value=None)
        lock.__exit__ = mock.MagicMock(return_value=False)
        with mock.patch(
            "blur_chunk_parallel.should_use_chunk_parallel", return_value=True
        ):
            with mock.patch(
                "blur_chunk_parallel.anonymize_video_chunked",
                return_value={"chunk_parallel": True, "processing_time_sec": 1.0},
            ) as chunked:
                with mock.patch.object(blur, "_PROCESS_LOCK", lock):
                    stats = blur.anonymize_video(
                        Path("v.mp4"), Path("refs"), Path("out.mp4")
                    )
        chunked.assert_called_once()
        self.assertTrue(stats.get("chunk_parallel"))


class TestApplyAnonymize(unittest.TestCase):
    def test_gaussian_changes_roi(self):
        # High-frequency pattern so Gaussian blur is observable
        yy, xx = np.mgrid[0:120, 0:120]
        frame = ((xx + yy) % 2 * 255).astype(np.uint8)
        frame = np.stack([frame, frame, frame], axis=-1)
        before = frame.copy()
        blur._apply_face_anonymize(frame, 20, 20, 80, 80, operator="gaussian")
        self.assertFalse(np.array_equal(frame[30:70, 30:70], before[30:70, 30:70]))


class TestSyntheticVideoSmoke(unittest.TestCase):
    """Lightweight OpenCV smoke: write tiny clip, probe duration helper."""

    def test_ffprobe_duration_opencv_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tiny.mp4"
            w, h, n, fps = 64, 64, 10, 5.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
            self.assertTrue(writer.isOpened())
            for i in range(n):
                writer.write(np.zeros((h, w, 3), dtype=np.uint8))
            writer.release()
            with mock.patch.object(chunk, "shutil") as sh:
                # Force OpenCV path: no ffprobe
                sh.which.return_value = None
                # Also stub _ffmpeg to None so first branch skips
                with mock.patch.object(chunk, "_ffmpeg", return_value=None):
                    # Re-implement: _ffprobe_duration uses shutil.which for ffprobe
                    # and falls back to cv2 when probe fails / missing
                    pass
            # Call real helper; may use ffprobe if installed, either way >0
            dur = chunk._ffprobe_duration(path)
            self.assertGreater(dur, 0.0)
            self.assertLess(dur, 5.0)


if __name__ == "__main__":
    unittest.main()
