import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from terra.maps_buffer import load_maps_from_disk


class MapsBufferDistanceContractTest(unittest.TestCase):
    @staticmethod
    def write_map(root: Path, distance=None):
        for folder in ("images", "occupancy", "dumpability", "distance"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        target = np.zeros((64, 64), dtype=np.int8)
        target[30:34, 30:34] = -1
        np.save(root / "images" / "img_1.npy", target)
        np.save(
            root / "occupancy" / "img_1.npy",
            np.zeros_like(target, dtype=np.int8),
        )
        np.save(
            root / "dumpability" / "img_1.npy",
            np.ones_like(target, dtype=np.int8),
        )
        if distance is not None:
            np.save(root / "distance" / "img_1.npy", distance)

    def test_missing_distance_fails_loudly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_map(root)
            with patch.dict(os.environ, {"DATASET_SIZE": "1"}):
                with self.assertRaisesRegex(
                    RuntimeError, "Missing required distance map"
                ):
                    load_maps_from_disk(str(root))

    def test_nonfinite_or_unnormalized_distance_fails_loudly(self):
        for value, message in (
            (np.nan, "non-finite"),
            (1.1, r"normalized to \[0, 1\]"),
        ):
            with self.subTest(value=value):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    distance = np.zeros((64, 64), dtype=np.float32)
                    distance[0, 0] = value
                    self.write_map(root, distance)
                    with patch.dict(
                        os.environ, {"DATASET_SIZE": "1"}
                    ):
                        with self.assertRaisesRegex(RuntimeError, message):
                            load_maps_from_disk(str(root))

    def test_wrong_shape_or_dtype_fails_loudly(self):
        cases = (
            (
                np.zeros((32, 32), dtype=np.float32),
                "shape mismatch",
            ),
            (
                np.zeros((64, 64), dtype=np.int16),
                "floating dtype",
            ),
        )
        for distance, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    self.write_map(root, distance)
                    with patch.dict(
                        os.environ, {"DATASET_SIZE": "1"}
                    ):
                        with self.assertRaisesRegex(RuntimeError, message):
                            load_maps_from_disk(str(root))


if __name__ == "__main__":
    unittest.main()
