#!/usr/bin/env python3
"""Classify T1 u85,000 failure modes from the archived per-step probe traces."""
import json
import numpy as np
from pathlib import Path

R = Path("/home/lorenzo/moleworks/.worktrees/terra_trench_fresh_dig_alignment_20260818/"
         "tools/trench_align_pilot_u85000_receipts")
DO, DO_NOTHING = 6, 7
WORST = ["trn-seg3-side2", "trn-net3-side2-s", "trn-net3-side1-road", "trn-seg2-side2"]


def load(arm):
    z = np.load(R / f"probe_{arm}_u085000.npz")
    j = json.loads((R / f"probe_{arm}_u085000.json").read_text())
    return z, j


def analyse(arm):
    z, j = load(arm)
    per_slot = j["per_slot"]
    n = len(per_slot)
    active = z["active"]                # (450, n)
    action = z["action"]
    dug = z["fresh_trench_cells_dug"]
    loaded = z["loaded"]
    appl = z["align_applicable"]
    valid = z["align_valid"]
    eff = z["action_had_effect"]
    pv = z["pose_valid_axis_count"]
    rows = []
    for k in range(n):
        a = active[:, k]
        L = int(a.sum())
        succ = bool(per_slot[k]["succeeded"])
        d = dug[:, k] * a
        dig_steps = np.flatnonzero(d > 0)
        last_dig = int(dig_steps[-1]) + 1 if dig_steps.size else 0
        # tail = the part of the episode after the last productive dig
        tail = L - last_dig
        tail_slice = slice(last_dig, L)
        ld = loaded[tail_slice, k]
        # loaded-at-end (cannot move while loaded => absorbing dump block)
        end_loaded = int(loaded[L - 1, k]) if L else 0
        loaded_tail_frac = float((ld > 0).mean()) if tail > 0 else 0.0
        # applicable+admissible opportunities in the tail that were not taken
        tail_appl = appl[tail_slice, k]
        tail_valid = valid[tail_slice, k]
        tail_admissible_opps = int(np.sum(tail_appl & tail_valid))
        tail_appl_total = int(tail_appl.sum())
        tail_do = int(np.sum(action[tail_slice, k] == DO))
        tail_donothing = int(np.sum(action[tail_slice, k] == DO_NOTHING))
        tail_noeff = float((~eff[tail_slice, k]).mean()) if tail > 0 else 0.0
        # cycle detection: is the tail action sequence periodic with small period?
        seq = action[tail_slice, k]
        period = 0
        if tail >= 20:
            for p in range(1, 13):
                q = seq[-min(tail, 120):]
                if len(q) > 2 * p and np.array_equal(q[p:], q[:-p]):
                    period = p
                    break
        rows.append(dict(
            slot=per_slot[k]["slot_index"], cond=per_slot[k]["primary_cell"],
            succ=succ, length=L, last_dig=last_dig, tail=tail,
            end_loaded=end_loaded, loaded_tail_frac=loaded_tail_frac,
            tail_admissible_opps=tail_admissible_opps, tail_appl=tail_appl_total,
            tail_do=tail_do, tail_donothing=tail_donothing, tail_noeff=tail_noeff,
            period=period,
            invalid=per_slot[k]["invalid_do_steps"],
            appl_do=per_slot[k]["fresh_applicable_do_steps"],
            succ_digs=per_slot[k]["successful_fresh_dig_steps"],
        ))
    return rows, j


rows, j = analyse("t1")
fails = [r for r in rows if not r["succ"]]
print(f"T1 u85k: {len(rows)} slots, {len(fails)} failures, "
      f"{sum(r['succ'] for r in rows)} successes")
print()


def classify(r):
    """(i) horizon-limited w/ steady progress, (ii) absorbing/stuck,
    (iii) dump-blocked, (iv) progress stopped but agent still active/free."""
    if r["end_loaded"] > 0:
        return "iii_dump_blocked_loaded_at_horizon"
    if r["tail"] <= 40:
        return "i_horizon_limited_progress_to_the_end"
    if r["loaded_tail_frac"] > 0.5:
        return "iii_dump_blocked_mostly_loaded_tail"
    if r["period"] > 0:
        return "ii_absorbing_cycle_period_%d" % r["period"]
    if r["tail_noeff"] > 0.8:
        return "ii_absorbing_no_effect"
    return "iv_progress_stopped_free_and_moving"


from collections import Counter
print("=== all 65 T1 failures ===")
c = Counter(classify(r) for r in fails)
for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
    print(f"  {v:3d} ({100*v/len(fails):5.1f}%)  {k}")
print()
for cond in WORST:
    g = [r for r in fails if r["cond"] == cond]
    tot = [r for r in rows if r["cond"] == cond]
    print(f"=== {cond}: {len(g)} failures / {len(tot)} slots ===")
    c = Counter(classify(r) for r in g)
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f"  {v:3d}  {k}")
    print("   median last_dig_step={:.0f} median tail={:.0f} "
          "mean tail_admissible_opps={:.1f} mean tail_appl={:.1f} "
          "mean tail_DO={:.1f} mean tail_DO_NOTHING={:.1f} mean tail_noeff={:.2f}".format(
              np.median([r["last_dig"] for r in g]), np.median([r["tail"] for r in g]),
              np.mean([r["tail_admissible_opps"] for r in g]),
              np.mean([r["tail_appl"] for r in g]),
              np.mean([r["tail_do"] for r in g]),
              np.mean([r["tail_donothing"] for r in g]),
              np.mean([r["tail_noeff"] for r in g])))
    print()

print("=== aggregate over all 65 failures ===")
for key in ("last_dig", "tail", "loaded_tail_frac", "tail_admissible_opps",
            "tail_appl", "tail_do", "tail_donothing", "tail_noeff", "period"):
    v = np.array([r[key] for r in fails], dtype=float)
    print(f"  {key:22s} mean={v.mean():8.2f} median={np.median(v):8.2f} "
          f"p90={np.percentile(v,90):8.2f} max={v.max():8.2f}")
print("  failures with end_loaded>0:", sum(1 for r in fails if r["end_loaded"] > 0))
print("  failures with any admissible opportunity in the tail:",
      sum(1 for r in fails if r["tail_admissible_opps"] > 0))
print("  failures whose tail has zero applicable states:",
      sum(1 for r in fails if r["tail_appl"] == 0))
print("  failures that ran >=90% of the episode after the last dig:",
      sum(1 for r in fails if r["tail"] > 0.9 * r["length"]))

# successes for contrast
succ = [r for r in rows if r["succ"]]
print()
print("=== successes, for contrast ===")
print("  median episode length:", float(np.median([r['length'] for r in succ])),
      " p90:", float(np.percentile([r['length'] for r in succ], 90)),
      " max:", max(r['length'] for r in succ))
print("  median digs:", float(np.median([r['succ_digs'] for r in succ])))

print()
print("################ deeper: what happens inside the tail ################")


def tail_detail(arm):
    z, j = load(arm)
    ps = j["per_slot"]
    active, action = z["active"], z["action"]
    dug, loaded = z["fresh_trench_cells_dug"], z["loaded"]
    appl, valid, pv = z["align_applicable"], z["align_valid"], z["pose_valid_axis_count"]
    fresh_cells = z["fresh_trench_cells"]
    out = []
    for k in range(len(ps)):
        a = active[:, k]
        L = int(a.sum())
        d = dug[:, k] * a
        ds = np.flatnonzero(d > 0)
        last = int(ds[-1]) + 1 if ds.size else 0
        sl = slice(last, L)
        n = max(L - last, 1)
        ap, va = appl[sl, k], valid[sl, k]
        out.append(dict(
            cond=ps[k]["primary_cell"], succ=bool(ps[k]["succeeded"]),
            tail=L - last,
            frac_applicable=float(ap.mean()),
            frac_admissible=float((ap & va).mean()),
            frac_appl_but_gate_would_block=float((ap & ~va).mean()),
            frac_loaded=float((loaded[sl, k] > 0).mean()),
            frac_do=float((action[sl, k] == DO).mean()),
            do_when_admissible=float(np.sum((action[sl, k] == DO) & ap & va)),
            do_when_blocked=float(np.sum((action[sl, k] == DO) & ap & ~va)),
            admissible_steps=int(np.sum(ap & va)),
            blocked_steps=int(np.sum(ap & ~va)),
            mean_fresh_in_cone=float(fresh_cells[sl, k].mean()),
        ))
    return out


det = tail_detail("t1")
f = [r for r in det if not r["succ"]]
print(f"T1 failures n={len(f)}  (tail = steps after the last productive dig)")
for key in ("tail", "frac_applicable", "frac_admissible",
            "frac_appl_but_gate_would_block", "frac_loaded", "frac_do"):
    v = np.array([r[key] for r in f])
    print(f"  {key:32s} mean={v.mean():7.3f} median={np.median(v):7.3f}")
print("  total tail steps where a DO would have been ADMITTED:",
      sum(r["admissible_steps"] for r in f))
print("  total tail steps where a DO was applicable but the gate would BLOCK:",
      sum(r["blocked_steps"] for r in f))
print("  DO actions actually taken at admitted steps:", sum(r["do_when_admissible"] for r in f))
print("  DO actions actually taken at blocked steps:  ", sum(r["do_when_blocked"] for r in f))
print("  failures whose tail has >=1 admitted step:",
      sum(1 for r in f if r["admissible_steps"] > 0), "/", len(f))
print("  failures whose tail is ENTIRELY inadmissible when applicable:",
      sum(1 for r in f if r["blocked_steps"] > 0 and r["admissible_steps"] == 0))
print()
for cond in WORST:
    g = [r for r in f if r["cond"] == cond]
    print(f"  {cond:22s} n={len(g):2d} adm_steps={sum(r['admissible_steps'] for r in g):5d} "
          f"blocked_steps={sum(r['blocked_steps'] for r in g):5d} "
          f"DO@adm={int(sum(r['do_when_admissible'] for r in g)):4d} "
          f"DO@blocked={int(sum(r['do_when_blocked'] for r in g)):4d} "
          f"frac_loaded={np.mean([r['frac_loaded'] for r in g]):.3f}")

print()
print("=== C0 contrast (9 failures) ===")
rows_c0, j0 = analyse("c0")
f0 = [r for r in rows_c0 if not r["succ"]]
c0 = Counter(classify(r) for r in f0)
for k, v in sorted(c0.items(), key=lambda kv: -kv[1]):
    print(f"  {v:3d}  {k}")
s0 = [r for r in rows_c0 if r["succ"]]
print("  C0 success median length:", float(np.median([r['length'] for r in s0])),
      " max:", max(r['length'] for r in s0))
print("  C0 failure median last_dig:", float(np.median([r['last_dig'] for r in f0])),
      " median tail:", float(np.median([r['tail'] for r in f0])))
