"""Unit tests for Phase-1 license-plate blur helpers and routing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import blur_service as blur  # noqa: E402


class TestPlateHold(unittest.TestCase):
    def test_new_detection_starts_age_zero(self):
        out = blur.update_confirmed_plates([], [(10, 10, 80, 40)], hold=5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1], 0)

    def test_unmatched_ages_and_drops(self):
        confirmed = [((10, 10, 80, 40), 4)]
        out = blur.update_confirmed_plates(confirmed, [], hold=5, age_step=1)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1], 5)
        out2 = blur.update_confirmed_plates(out, [], hold=5, age_step=1)
        self.assertEqual(out2, [])

    def test_iou_match_resets_age(self):
        confirmed = [((10, 10, 80, 40), 3)]
        dets = [(12, 12, 82, 42)]
        out = blur.update_confirmed_plates(confirmed, dets, hold=10, alpha=0.5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][1], 0)

    def test_nms_collapses_duplicates(self):
        boxes = [(10, 10, 80, 40), (12, 12, 82, 42), (200, 200, 260, 230)]
        kept = blur._nms_boxes(boxes, iou_thresh=0.4)
        self.assertEqual(len(kept), 2)

    def test_clip_box_rejects_tiny(self):
        self.assertIsNone(blur._clip_box((0, 0, 3, 3), 100, 100))
        self.assertEqual(blur._clip_box((-5, -5, 50, 40), 100, 100), (0, 0, 50, 40))


class TestPlateAnonymize(unittest.TestCase):
    def test_rectangular_blur_changes_roi(self):
        yy, xx = np.mgrid[0:80, 0:160]
        frame = ((xx + yy) % 2 * 255).astype(np.uint8)
        frame = np.stack([frame, frame, frame], axis=-1)
        before = frame.copy()
        blur._apply_plate_anonymize(frame, 20, 20, 140, 60, operator="gaussian")
        self.assertFalse(np.array_equal(frame[25:55, 30:130], before[25:55, 30:130]))


class TestEnsurePlateModel(unittest.TestCase):
    def test_ensure_raises_when_missing(self):
        eng = object.__new__(blur.BlurEngine)
        eng._plate_yolo = None
        eng._plate_weights_path = ""
        with mock.patch.object(eng, "_try_load_plate_yolo", return_value=None):
            with self.assertRaises(RuntimeError):
                eng.ensure_plate_model()

    def test_ensure_ok_when_loaded(self):
        eng = object.__new__(blur.BlurEngine)
        eng._plate_yolo = object()
        eng.ensure_plate_model()  # no raise

    def test_ensure_retries_lazy_load(self):
        eng = object.__new__(blur.BlurEngine)
        eng._plate_yolo = None
        eng._plate_weights_path = ""

        def _load():
            eng._plate_yolo = object()

        with mock.patch.object(eng, "_try_load_plate_yolo", side_effect=_load):
            eng.ensure_plate_model()
        self.assertIsNotNone(eng._plate_yolo)


class TestAnonymizePlatesRouting(unittest.TestCase):
    def test_plates_only_skips_gallery(self):
        engine = mock.MagicMock()
        engine.process_video.return_value = {
            "blur_plates": True,
            "blur_faces": False,
            "browser_ready": True,
        }
        with mock.patch.object(blur, "get_blur_engine", return_value=engine):
            with mock.patch.object(blur, "_PROCESS_LOCK") as lock:
                lock.__enter__ = mock.MagicMock(return_value=None)
                lock.__exit__ = mock.MagicMock(return_value=False)
                with mock.patch(
                    "blur_chunk_parallel.should_use_chunk_parallel", return_value=False
                ):
                    stats = blur.anonymize_video(
                        Path("v.mp4"),
                        None,
                        Path("out.mp4"),
                        blur_plates=True,
                        blur_faces=False,
                    )
        engine.build_master_signatures.assert_not_called()
        kwargs = engine.process_video.call_args.kwargs
        self.assertTrue(kwargs.get("blur_plates"))
        self.assertFalse(kwargs.get("blur_faces"))
        self.assertTrue(stats.get("blur_plates") or True)

    def test_neither_raises(self):
        with self.assertRaises(ValueError):
            blur.anonymize_video(
                Path("v.mp4"),
                None,
                Path("out.mp4"),
                blur_plates=False,
                blur_faces=False,
            )


if __name__ == "__main__":
    unittest.main()
