#!/usr/bin/env bash
cd /home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818
R=tools/trench_align_v2_maxoffset_sweep_20260902
WORKERS=${WORKERS:-22} BOUNDS="off:0 b114:1.14 b171:1.71 b229:2.29 b286:2.86 b343:3.43" \
  PHASES="pooled preflight" bash $R/run_sweep.sh
WORKERS=${WORKERS:-22} BOUNDS="b115:1.15 b172:1.72" \
  PHASES="cover overres axis pooled" bash $R/run_sweep.sh
echo ALL_PHASES_DONE > $R/sweep_all.done
