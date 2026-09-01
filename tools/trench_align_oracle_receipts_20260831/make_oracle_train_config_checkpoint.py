"""Synthesize a minimal T1-arm checkpoint: only train_config is read by the oracle."""
import pickle, sys
from pathlib import Path
sys.path.insert(0, "/home/lorenzo/moleworks/.worktrees/terra_baselines_trench_pose_alignment_20260818")
from train import TrainConfig  # noqa
from train_mixed import MixedAgentTrainConfig  # noqa

cfg = MixedAgentTrainConfig(
    num_devices=1,
    num_envs_per_device=176,
    name="trench_align_t1_v1",
    num_prev_actions=5,
    agent_types_override=(0,),
    action_types_override=(0,),
    relocation_progress_mult=1.5,
    reward_stage="reward_v2",
    distance_protocol_id="obstacle_geodesic_8_physical_global_v1",
    carry_work_observation=True,
    stall_age_observation=False,
    reward_v2_reset_context_observation=False,
    trench_alignment_observation=True,
    require_trench_alignment_metadata=True,
    enforce_trench_dig_alignment=True,
    trench_dig_standoff_enforced=True,
    model_core="mlp",
    model_size="medium",
    map_encoder="resnet_spatial_8x8_se_sa_xattn",
    critic_hidden_dims=(512, 256),
    encoder_compute_dtype="bfloat16",
    attention_compute_dtype="float32",
    token_mixer_residual_init_scale=0.1,
    flatten_reduce_channels=32,
    attn_latent_queries=8,
    aux_coef=0.0,
    config_name="trench_align_t1_v1",
)
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("wb") as fh:
    pickle.dump({"train_config": cfg}, fh)
print("wrote", out)
