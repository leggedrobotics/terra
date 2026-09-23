#!/usr/bin/env python3
"""Load the pooled bank through the trainer's own path on CPU.

Mirrors ``train_mixed.py``: the ``trench_align_v2_generalist_gen`` preset's map
level (maps_path swapped to the new pooled folder) -> ``BatchConfig`` ->
``TerraEnvBatch(distance_protocol_id=R2)`` (``init_maps_buffer`` ->
``load_maps_from_disk`` with the exact-contract and R2 checks) ->
``create_mixed_agent_env_config`` -> ``_preflight_trench_alignment_metadata``
(finite trench sections for every map, and Terra's array-level validator).
Requires DATASET_PATH / DATASET_SIZE and PYTHONPATH=<terra>:<terra-baselines>.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

BASELINES = Path(os.environ["BASELINES_ROOT"]).resolve()
sys.path.insert(0, str(BASELINES))
os.chdir(BASELINES)

from configs.training_configs import get_config  # noqa: E402
from terra.config import BatchConfig, CurriculumGlobalConfig, RewardsType  # noqa: E402
from terra.env import TerraEnvBatch  # noqa: E402
import train_mixed as tm  # noqa: E402


def main() -> None:
    maps_path = sys.argv[1]
    output = Path(sys.argv[2])
    preset = get_config("trench_align_v2_generalist_gen")
    levels = levels_ = [
        {
            "maps_path": maps_path,
            "max_steps_in_episode": level.max_steps_in_episode,
            "rewards_type": RewardsType.DENSE if level.rewards_type == "DENSE" else RewardsType.SPARSE,
            "apply_trench_rewards": level.apply_trench_rewards,
        }
        for level in preset.maps
    ]

    increase_th = preset.curriculum.increase_level_threshold
    decrease_th = preset.curriculum.decrease_level_threshold
    last_level = preset.curriculum.last_level_type

    class CustomCurriculumGlobalConfig(CurriculumGlobalConfig):  # as make_mixed_agent_states
        levels = levels_
        increase_level_threshold = increase_th
        decrease_level_threshold = decrease_th
        last_level_type = last_level

    batch_cfg = BatchConfig(curriculum_global=CustomCurriculumGlobalConfig())
    started = time.time()
    env = TerraEnvBatch(
        batch_cfg=batch_cfg,
        shuffle_maps=False,
        distance_protocol_id="obstacle_geodesic_8_physical_global_v1",
        movement_feasibility_observation=False,
        previous_outcome_observation=False,
        executable_dig_observation=False,
    )
    load_seconds = time.time() - started
    leaves = jax.tree_util.tree_leaves(env.maps_buffer)
    array_bytes = int(sum(np.asarray(leaf).nbytes for leaf in leaves if hasattr(leaf, "shape")))
    shapes = {
        name: list(np.shape(getattr(env.maps_buffer, name)))
        for name in env.maps_buffer._fields
        if hasattr(getattr(env.maps_buffer, name), "shape")
    }
    env_params = tm.create_mixed_agent_env_config(
        agent_types=tuple(preset.agent_types),
        action_types=tuple(preset.action_types),
        relocation_progress_mult=preset.relocation_progress_mult,
        reward_stage="reward_v2",
        enforce_trench_dig_alignment=preset.enforce_trench_dig_alignment,
        trench_dig_standoff_enforced=preset.trench_dig_standoff_enforced,
        trench_dig_max_offset_m=preset.trench_dig_max_offset_m,
    )
    started = time.time()
    tm._preflight_trench_alignment_metadata(env, env_params, levels)
    preflight_seconds = time.time() - started
    report = {
        "dataset_path": os.environ["DATASET_PATH"],
        "dataset_size": int(os.environ["DATASET_SIZE"]),
        "maps_path": maps_path,
        "preset": "trench_align_v2_generalist_gen (maps_path swapped)",
        "host_load_seconds_terra_env_batch": round(load_seconds, 1),
        "trainer_metadata_preflight_seconds": round(preflight_seconds, 1),
        "maps_buffer_array_bytes": array_bytes,
        "maps_buffer_shapes": shapes,
        "jax_backend": jax.default_backend(),
        "passed": True,
    }
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
