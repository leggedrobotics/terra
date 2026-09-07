import unittest
from types import SimpleNamespace

import numpy as np

from terra.config import EnvConfig
from terra.env import TerraEnvBatch


class TrenchMetadataValidationTests(unittest.TestCase):
    @staticmethod
    def batch(family_names, family_ids, trench_types):
        batch = object.__new__(TerraEnvBatch)
        types = np.asarray([trench_types], dtype=np.int32)
        records = np.full((*types.shape, 4, 8), -97.0, dtype=np.float32)
        for slot, count in enumerate(trench_types):
            if count > 0:
                records[0, slot, 0] = [0.0, 1.0, -32.0, 32.0, 20.0, 32.0, 44.0, 1.0]
        batch.maps_buffer = SimpleNamespace(
            family_names=family_names,
            family_ids=np.asarray([family_ids], dtype=np.int32),
            trench_types=types,
            trench_axes=records,
        )
        return batch

    def validate(self, batch):
        batch._validate_trench_alignment_metadata_requirements(
            EnvConfig(enforce_trench_dig_alignment=True)
        )

    def test_explicit_foundations_need_no_trench_axes(self):
        batch = self.batch(
            ("unknown", "foundation", "fnd", "fnd-square"),
            [1, 2, 3], [-1, -1, -1],
        )
        self.validate(batch)

    def test_unknown_maps_without_axes_are_rejected_including_mixed_banks(self):
        for names, ids, types in (
            (("unknown",), [0], [-1]),
            (("foundation",), [-1], [-1]),
            (("foundation", "unknown"), [0, 1], [-1, -1]),
            (("trench", "unknown"), [0, 1], [1, -1]),
        ):
            with self.subTest(names=names, ids=ids):
                with self.assertRaisesRegex(RuntimeError, "family provenance"):
                    self.validate(self.batch(names, ids, types))

    def test_missing_trench_axes_are_rejected_including_mixed_banks(self):
        for names, ids, types in (
            (("foundation", "trench"), [0, 1], [-1, -1]),
            (("foundation", "trn-straight"), [0, 1, 1], [-1, 1, -1]),
        ):
            with self.subTest(names=names):
                with self.assertRaisesRegex(RuntimeError, "trench-family maps lack axis metadata"):
                    self.validate(self.batch(names, ids, types))

    def test_valid_mixed_bank_keeps_finite_section_validation(self):
        batch = self.batch(("foundation", "trench"), [0, 1], [-1, 1])
        self.validate(batch)
        batch.maps_buffer.trench_axes[0, 1, 0, 3:7] = -97.0
        with self.assertRaisesRegex(RuntimeError, "finite section metadata"):
            self.validate(batch)


if __name__ == "__main__":
    unittest.main()
