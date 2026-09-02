#!/usr/bin/env bash
# Scripted-oracle solvability run at one v2 "on the line" bound.
#   run_oracle.sh <tag> <max_offset_m>
# tag "off" (bound 0) reproduces the 2026-09-01 yaw-only v2 run (146/176).
set -eu
TAG=$1
M=$2
W=/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818
BASE=/home/lorenzo/moleworks/.worktrees/terra_baselines_trench_pose_alignment_20260818
PY=/home/lorenzo/moleworks/.venv-terra-uv/bin/python
BANK=/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819
R=$W/tools/trench_align_v2_maxoffset_sweep_20260902
export JAX_PLATFORMS=cpu PYTHONPATH=$W:$BASE
cd $BASE
$PY scripts/trench_align_scripted_oracle.py \
  --checkpoint $W/tools/trench_align_oracle_receipts_20260831/oracle_t1_arm_train_config_only.pkl \
  --bank-root $BANK \
  --terra-revision a6e6e5bc1cd29e4f3a5c8d99a7fbd9fe855ba1b4 \
  --panel-family gate_main --accepted-panel development \
  --horizon 450 --extended-horizon 900 --verify-action-mask \
  --max-offset-m "$M" \
  --output $R/oracle_176slot_$TAG.json 2>&1 | tee $R/oracle_176slot_$TAG.log
