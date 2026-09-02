#!/usr/bin/env bash
# Coverage sweep of the v2 "on the line" bound (EnvConfig.trench_dig_max_offset_m).
# Every job is the same tool as the 2026-09-01 v2 revalidation, with --max-offset-m
# added; 0 disables the clause (yaw-parallel only) and reproduces that run.
set -u
W=/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818
PY=/home/lorenzo/moleworks/.venv-terra-uv/bin/python
BANK=/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819
R=$W/tools/trench_align_v2_maxoffset_sweep_20260902
WORKERS=${WORKERS:-28}
export JAX_PLATFORMS=cpu PYTHONPATH=$W
cd $W
LOG=$R/run_log.txt

run () {  # run <name> <cmd...>
  local name=$1; shift
  if [ -s "$R/$name.done" ]; then echo "SKIP $name" | tee -a $LOG; return 0; fi
  echo "=== START $name $(date -Iseconds) ===" | tee -a $LOG
  local t0=$SECONDS
  "$@" > "$R/$name.log" 2>&1
  local rc=$?
  echo "=== END   $name rc=$rc $((SECONDS-t0))s $(date -Iseconds) ===" | tee -a $LOG
  [ $rc -eq 0 ] && echo ok > "$R/$name.done"
  return $rc
}

PREFLIGHT_DATASETS=(
  --dataset train/018__trn-net3-side1-road --dataset train/019__trn-net3-side2
  --dataset train/020__trn-net3-side2-s --dataset train/021__trn-net4-side1-road
  --dataset train/022__trn-net4-side2 --dataset train/023__trn-net4-side2-s
  --dataset train/024__trn-seg2-side2 --dataset train/025__trn-seg3-side2
  --dataset train/026__trn-straight-altsides --dataset train/027__trn-straight-side1
  --dataset train/028__trn-straight-side1-tight --dataset train/029__trn-straight-side2
  --dataset train/030__trn-tee-side2 --dataset train/031__trn-tee-side2-s
  --dataset train/033__trn-straight-allfree
  --dataset evaluation/gate_main/development --dataset evaluation/capability_floor/development
  --dataset evaluation/gate_main/promotion --dataset evaluation/capability_floor/promotion
  --dataset evaluation/gate_main/sealed --dataset evaluation/capability_floor/sealed
)

# tag -> bound in metres (0 = clause disabled, i.e. the yaw-only v2 baseline)
BOUNDS=${BOUNDS:-"off:0 b114:1.14 b171:1.71 b229:2.29 b286:2.86 b343:3.43"}

PHASES=${PHASES:-"cover overres axis pooled preflight"}

for phase in $PHASES; do
for pair in $BOUNDS; do
  tag=${pair%%:*}; m=${pair##*:}
  case $phase in
  cover)
    run "station_cover_gate_main_dev_$tag" \
      $PY tools/check_trench_persistent_station_cover.py --bank-root $BANK \
      --dataset evaluation/gate_main/development --exclude-prefix \
      --workers $WORKERS --max-offset-m $m \
      --output $R/station_cover_gate_main_dev_$tag.json ;;
  overres)
    run "overrestriction_gate_main_dev_$tag" \
      $PY tools/audit_trench_gate_overrestriction.py --bank-root $BANK \
      --dataset evaluation/gate_main/development --exclude-prefix \
      --workers $WORKERS --max-offset-m $m \
      --output $R/overrestriction_gate_main_dev_$tag.json ;;
  axis)
    run "axis_sweep_gate_main_dev_$tag" \
      $PY tools/check_trench_axis_sweep_feasibility.py --bank-root $BANK \
      --dataset evaluation/gate_main/development --exclude-prefix \
      --workers $WORKERS --max-offset-m $m \
      --output $R/axis_sweep_gate_main_dev_$tag.json ;;
  pooled)
    run "overrestriction_train_v2_pooled_$tag" \
      $PY tools/audit_trench_gate_overrestriction.py --bank-root $BANK \
      --dataset train_v2_pooled_generalist --exclude-prefix \
      --workers $WORKERS --max-offset-m $m \
      --output $R/overrestriction_train_v2_pooled_$tag.json ;;
  preflight)
    run "preflight_full_$tag" \
      $PY tools/audit_trench_alignment_feasibility.py --bank $BANK --layout exact \
      "${PREFLIGHT_DATASETS[@]}" --workers $WORKERS --witness-actions counts \
      --max-offset-m $m --output $R/preflight_full_$tag.json ;;
  esac
done
done
echo "ALL DONE $(date -Iseconds)" | tee -a $LOG
