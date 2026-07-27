# Terra Training Tasks

- Status: active recovery execution; semantics and integrity gates C0-C5/C1b
  are complete; F0/F0R established single-identity feasibility; B0a static
  panels and all bounded-run integrity gates pass. B0 foundation-distance
  diversity shows real source-disjoint progress and authorizes one fresh
  5,000-update confirmation, but has no adjacent cell witness. Foundation
  geometry remains 0/8 on both held-out cells after 5,000 updates despite one
  small procedural-completion gain. The exhausted one-side trench isolate
  reaches roughly 96% sampled-train success but peaks at only 4/8 held out and
  ends 2/8, so it stops for generalization diagnosis. No family, B0c, F1, or
  `gpuhe.120h` run is qualified.
- Date: 2026-07-27 execution update
- Governing design: [`TRAINING_DESIGN.md`](TRAINING_DESIGN.md)
- Canonical current map-benchmark/curriculum specification:
  [`MAP_BENCHMARK_SPEC.md`](MAP_BENCHMARK_SPEC.md), especially the v0.3
  reviewer decision log in Section 20
- Active first subgoal: standardize the training distribution and deliver its
  graphical review workflow under
  [`DIGGING_BENCHMARK_SITE_GOAL.md`](DIGGING_BENCHMARK_SITE_GOAL.md); it is
  jointly governed by this ledger and `MAP_BENCHMARK_SPEC.md`
- Failure evidence: [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md)
- Historical reference: E8 `resnet_spatial_8x8_se`
- Recovery scratch topology: base `resnet_spatial_8x8`, approximately 994,825
  parameters
- Production training authorized by this document: yes, only for declared
  tasks whose gates pass; independent declared arms may run concurrently

The latest accepted map-curriculum design must be written into
`MAP_BENCHMARK_SPEC.md` in the same change that records its reviewer
disposition. This execution ledger may summarize that design but must not
silently override it. Chat history is not an execution dependency.

## 1. Current decision

As of 2026-07-27, the task-semantics recovery is complete but no broad map
curriculum is selected. Corrected foundation and trench runs are numerically
healthy and prove partial feasibility; their source-disjoint evaluations show
memorization, non-monotonic cell difficulty, and regression. More unchanged
PPO is not justified.

The live plan is the factorized admission graph in
[`MAP_BENCHMARK_SPEC.md`](MAP_BENCHMARK_SPEC.md), with its accepted-review
log in Section 20. The benchmark/site goal in
[`DIGGING_BENCHMARK_SITE_GOAL.md`](DIGGING_BENCHMARK_SITE_GOAL.md) is now the
first delivery priority. The immediate order is:

1. standardize the public review-data contract and serve the graphical
   inspector locally against current B0a data, with separately browsable
   depth/family/condition example folders and explicit design-input labels
   rather than an admitted-benchmark claim;
2. S1: complete state/condition schema, live-geometry revalidation,
   exact direct-service metric and cost receipt, pair-specific foundation
   support, topology-aware trench support, and adjustable apron capacity;
3. S2: replace the preview data with the balanced 448-scenario pilot and
   record Lorenzo's visual decisions;
4. S3: replay exact witnesses within 450 steps; and
5. only then materialize selected active cells and resume specialist/generalist
   PPO under source-disjoint promotion gates.

The rejected flat M0-M2 and per-environment 3/3 treatments and all completed
B0 evidence remain below as historical receipts. They do not authorize their
old forward-looking ladders.

### 1.1 Ratified decisions from the design review

These decisions supersede stale choices later in the historical v4 design:

| Topic | Ratified recovery decision |
|---|---|
| Legal dump region | The exact visible target dump mask is authoritative everywhere. There is no hidden one-cell buffer. |
| Starter soil physics | A correctly aimed dump is contained inside the exact target region and conserves mass. No boundary clipping or deletion is permitted. |
| Wrong dumps | An entirely off-zone dump remains physically possible, remains off-zone, earns no legal completion, and must be recovered. |
| Later spill difficulty | Physical boundary spill is deferred to a separately named dynamics treatment after contained-map competence. |
| Action validity | Remove every relocation-potential veto on a physically valid dump. Potential belongs in reward, not action legality. |
| Initialization | Train new small policies from scratch on the corrected distribution. E8 is evaluation context only, never the initializer or teacher for recovery runs. |
| Family separation | Foundation and trench feasibility/specialist policies are two independent runs. |
| Scratch budget | Treat 500/1,000/2,000/5,000 updates as review milestones, not hard ceilings. Any integrity-clean recipe that shows even slight preregistered fixed-bank improvement receives the next meaningful budget; use an exact continuation only when the checkpoint contract preserves all process state, otherwise run the full higher milestone continuously from a declared fresh start. |
| Qualified long runs | Each family specialist independently earns long training as soon as its own recipe clears the twice-observed source-disjoint gate; a qualified foundation recipe does not wait for trench, or vice versa. Run selected F1/G0/S0/K0 treatments for at least 20,000 continuous updates on `gpuhe.120h` with a five-day wall-time request. Grant more compute while the fixed bank improves, using exact 20,000-update extensions when available or a fresh continuous run at the full higher budget. A short-run wall-clock limit must never stop a recipe that has cleared this qualification. |
| Curriculum separation | Map, dense-reward, dense-to-terminal reward, and partial-reset treatments never advance in the same causal comparison. |
| Map display depths | Use Anchor, One-axis, and Composed with literal prerequisite condition IDs. Retire M0-M5 as an active total order. |
| Runtime geometry | Keep live `36.5714285714 / 64 = 0.571428571428125` m/tile, the derived `7 x 11` footprint, and current action workspace. Recompute stale B0 static receipts; never change `edge_length_m` to fit old metadata. |
| Separation versus rehandling | `d02`-`d08` is dig/dump separation, not loaded transport. Add action-reachable exact dig-to-dump direct-service coverage before admitting forced-rehandling maps; defer relay hops until their graph is defined. |
| Map-curriculum reset | Use 100% untouched 450-step full resets. The 25% partial-reset hypothesis remains PR0 after sampler selection. |
| Source handling | OSM/procedural is provenance conceptually but remains separately matched and gated until both source slices pass; B0-GEO-F completed with 0/8 in both cells at all 50 checkpoints. |
| First new generator | Build an exact-dig `slcap03_04` versus `slcap07_10` nearby-apron pair with matched separation before defining capacity progression; the token names single-layer accepted-area ratio. |
| Pilot scale | S2 uses four foundation and four trench conditions x `32 train + 8 promotion + 8 development + 8 sealed = 448` scenarios. The eighth cell is volume-matched segmented-3, not an unsupported high-volume candidate. Expand only selected active training conditions afterward. |
| Family/rehearsal sampling | Generalists preserve 50/50 foundation/trench sampling, then use 50% frontier/50% admitted rehearsal inside an active family. Specialists remain single-family. |
| Feasibility horizon | Pilot-ranked maps need an exact replay witness within 450 steps. Record margin; do not silently enforce an unratified 225-step Core gate. |
| Exact reset state | Hash every reset-consumed `Agent`/`AgentState` field. A seed or partial pose tuple is not a portable scenario state. |
| Trench volume support | `v68_77` is frozen from the train-only S1 audit using fixed radius-one straight, segmented-2 `U[10,13)`, and segmented-3 `U[7,9.5)` generation. Existing B0 trench identities do not carry into the pilot. |
| Foundation volume support | Freeze OSM/procedural all-around at train-audited `v140_189`, compactness `[0.30,0.65]`, and exact pairwise volume/perimeter after retuning only procedural main length/width. Match the apron capacity pair separately by exact OSM dig identity. |
| Retention arithmetic | Freeze the lower of two passing counts. Retain an 8-map condition at `max(6, reference-1)` and the fixed 32-map family panel at `max(26, reference-1)`; only consecutive complete integrity-valid failures count toward rollback. |
| Migration/cost evidence | Withdraw the unreceipted 23/256 probe count. S1 writes a hashed per-identity migration receipt and an exact direct-service validation cost profile before S2. |
| Pilot initial states | Materialize one hash-seeded live `Agent.new` result per source group and split over the variants' intersected spawn contract. Counterfactuals share the exact state; never resample after observing feasibility or policy performance. |
| Direct-service union | Replay actual movement/dig/cabin/dump transitions. Off-zone complete dumps remain recoverable mistakes but do not count; overlapping hypothetical digs are unioned by capped per-cell maximum progress. |
| B0a migration identity | Use legacy `source_id` as canonical `source_group_id`: 144 groups over 256 rows. Keep pair/topology-match labels separate; topology matches never share reset state. Gate reachable dump capacity, not every optional component, and keep direct-service coverage diagnostic. |

### 1.2 First S1 execution receipts

- [x] The deterministic train-only trench audit sampled 20,000 raster-valid
  proposals per topology under seed `2026072701`, selected the closed
  `v68_77` band, and reproduced byte-identically in an independent second
  output directory. Support was `21.46%` straight, `52.795%` segmented-2,
  and `79.50%` segmented-3; accepted unique-raster counts were 3,077, 8,130,
  and 15,420. The full length/turn/volume/width/rejection receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_pilot_trench_volume_support_20260727_v2/`.
  `support_summary.json` has SHA-256
  `a0927ef98e13774edfe0c500184ba1dd88b23e35bb67b14b0628c852bb8fb01c`;
  `samples.jsonl` has SHA-256
  `651e3487838175e8d6c23bc86994dfa55fdffe8b843dd2a804b18ca88dedda54`.
  The historical B0 builder remains byte-identical at SHA-256
  `3a1bb66798f6a4bfc1dc5b3515c5a4485eb9a28d6ffe7c9e8413a548492b79a9`.
- [x] The fixed-bank history aggregator now applies the ratified integer
  retention contract, including lower-of-two mastery references, per-panel
  streak reset, neutral invalid/incomplete evaluations, and a sticky
  two-failure rollback receipt. Focused `n=8`, `n=32`, evaluator, compile, and
  formatting gates pass in the paired terra-baselines worktree at commit
  `671c4d9`.
- [x] The tracked pilot generator now constructs `slcap03_04` and
  `slcap07_10` from one exact OSM dig raster/source group at nominal ratios
  `3.25` and `8.5`, hard-failing outside the closed capacity or `sep02` p50
  bands. Six focused tests pass. A read-only diagnostic constructed both
  variants for all 16 already-burned B0a train/development OSM apron sources;
  it selected no parameter and is not admission evidence. This closes the
  pure constructor subtask only: S1 Capacity remains open until a new
  train-only materialized receipt, live static validation, and Lorenzo's
  visual review pass.
- [x] The fresh train-only capacity review bank materializes `32` source-
  disjoint OSM pairs (`64` maps) with one exact dig raster, required volume,
  separation distribution, and reset state shared inside each pair. The low
  band spans `3.2500-3.2547x`, the high band `8.5000-8.5035x`, and all p50/p95
  high-minus-low separation deltas are exactly zero. All `427` manifested
  files and the independent verifier pass; `files.sha256` has SHA-256
  `03b5af27c3fc2acea05b6ecb001e4dcb7cdcbfe3f2677efa64ad2a6e3e7df80f`.
  The visual receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_pilot_apron_capacity_review_20260727_v1/`.
  Work is only `140-189` cells (`3.42%-4.61%` of the map), confirming R-42:
  this is a controlled capacity slice, not broad foundation-size coverage.
  Canonical Format admission, exact Static validation, Lorenzo's review
  decision, and S1 Capacity completion remain open.
- [x] Single and batched resets now accept a complete explicit `Agent` tree
  while preserving every leaf and the reset RNG. `terra_agent_state_v1`
  hashes all four live slots with path/dtype/rank/shape-prefixed canonical
  bytes, rejects untracked schema changes, and validates the one-active
  tracked-excavator full-reset contract with the live footprint and shared
  eight-tile border constant. The canonical test vector is
  `debd22b6ff2c8b31d263ceb843e524d5bf9ae1ffe186e26291f1e5ec3d18fb1a`;
  eager/JIT/batched-vmap preservation and the existing dump contract pass
  together (`22 passed`).
- [x] R-25's initial-state namespace and sampler are implemented: the first
  four SHA-256 bytes are interpreted big-endian as a `uint32`, one live
  tracked `Agent.new` call consumes the source-group-intersected spawn
  contract, and the receipt pins both seed and canonical state hashes. The
  golden namespace yields seed `1643655228` and digest
  `61f8303cdc0376bdf2d348c248f3cbd1a16678764f6e276affce135fa2463329`;
  deterministic and shared-counterfactual tests pass. B0a materialization is
  receipted below; materialization for the fresh S2 bank remains open.
- [x] The live B0a migration normalizer verifies all `3,392` frozen input
  files, derives the protocol from Terra, normalizes all `256` design-input
  identities, and materializes one exact serialized state for each of `144`
  canonical source groups. The raw `int8` raster identity and the exact-
  loader `int16` value view are checked separately after the first real run
  exposed that dtype boundary. Two clean runs at commit `affc0d92` are byte-
  identical: `migration_validation.jsonl` SHA-256
  `ce14b52e330cd734997f3c269b85d58a93ab24f67fc1e3f1499af0ecc5228b37`
  and `migration_summary.json` SHA-256
  `7ec113fa87705c239819162ce954b8ead014bc6c8d1d11dd2fb3ed8e8a57300f`.
  Every row passes affordable semantics, exact capacity, initial-state, and
  migration-record checks with no errors. Every row deliberately remains
  `benchmark_format_valid=false`, `static_valid=null`, and
  `pending_exact_static`; the receipt does not satisfy the blocked exact
  direct-service gate or admit an S2 condition.
- [x] The train-only foundation source audit freezes `v140_189`,
  compactness `[0.30,0.65]`, and exact pairwise required volume plus exposed
  four-neighbour perimeter. In the fixed 256-group audit-train partition,
  111 source-bank rasters are supported, 98 have at least two unique exact
  procedural candidates, and 32 unique pairs were selected. The retuned
  procedural sampler produced 3,732 supported proposals (3,721 unique) from
  20,000 after changing only its main length/width ranges to `U[13,22)` and
  `U[7,13)`. Reserve masks were used only for canonical identity and fixed
  factory eligibility; no reserve distribution/support/matching selection
  occurred. The byte-reproduced receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_pilot_foundation_source_support_20260727_v3/`;
  `support_summary.json` has SHA-256
  `e0867f67669c5f7f2e6ce15890c2c824783666056ddb58ebe21e56107191261f`
  and `matched_train_pairs.jsonl` has SHA-256
  `8c5528a2b43a485858df15acbca5835a41be8f9d3d8ab6159cf272c6f9d06688`.
  Cross-family canonical-raster equality is a hard error and Python `3.12.11`,
  NumPy `1.26.4`, and SciPy `1.12.0` are receipted. Residual bbox/moment aspect
  differences remain explicit audit covariates, so this is not called a
  source-only intervention. Raster support is closed; tracked generator
  portability is required before S2, and raw OSM feature attribution is still
  required before publication.
- [x] The exact initial direct-service validator now enumerates base poses
  reachable by Terra's real tracked forward/backward/base-rotation
  transitions, replays every cabin heading with the real dig/dump transition,
  and uses the exact visible accepted mask. Entirely off-zone complete dumps
  are diagnostic mistakes, mixed accepted/off-zone complete dumps hard-fail,
  and overlapping hypothetical digs are combined by capped per-cell maximum
  rather than summed. Logical attempts and padded JAX executions are
  separately receipted, and an exhaustive fixed-pose differential test guards
  the dig prefilter against false negatives. The combined direct-service and
  dump-contract suite passed `23` tests plus `4` transition-parity subtests in
  `358.01 s` wall time with `6,168,612 KiB` peak RSS on CPU. The implementation
  and focused-test SHA-256 values are
  `3469f3b04f4e66379aa52a9b0c8d2cd97f567590c1e52fd9d75da3004420b2c3`
  and
  `2dc3a7bb856240ff731ac976bd817f438db4cdb7668632e769359e972efbba15`.
  This closes the semantic implementation gate only; it is not the required
  64 x 64 cost receipt.
- [x] The fixed non-admission 64 x 64 cost probe completed on
  `b0a-train-f_apron_d02-00` from source group `osm-foundation:108`, with the
  four distance counterfactuals sharing explicit state SHA-256
  `225ad1df2a48a1b098fb2a095e9382185dded7516e2b79d5b98c7b70e7137c02`.
  It found 31,366 reachable base poses, 376,392 pose/cabin rows, and 39,606
  exact-service candidate rows. Cold/warm graph-plus-prefilter cost was
  `190.440/115.293 s`; service lowering/compile was `128.369/13.781 s`; and
  warmed four-row batch p50/p95 was `0.072704/0.073481 s`. The one-scenario
  p50/p95 projection is `1,052.509/1,060.204 s`, with process peak
  `2,946,580 KiB`, so the preregistered 60-minute and memory gates pass and
  authorize exactly one complete-scenario confirmation. The provisional
  256/448 p95 projections are `216,001/377,839 s` (about `60.0/105.0 h`), so
  neither larger run is authorized without calibration and likely
  optimization. The clean-worktree receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_cost_probe_20260727_v1/validation_cost_probe.json`
  at SHA-256
  `d283351640e5eaf2cc2ac5734766175a9863dc749bd4067a6c88c94f3d72e110`;
  it emits no admission or feasibility outcome.
- [x] The one authorized complete exact confirmation reproduced the same
  state and dependency hashes and made one
  `compute_initial_direct_service` call. Exact-validator wall time was
  `1,086.615 s` versus the probe's `1,052.509 s` p50, a ratio of `1.0324`;
  process peak was `3,558,344 KiB`, or `3.61%` of host capacity. The selected
  scenario has required volume `104`, workspace/direct-serviceable volume
  `104/104`, and direct-service coverage `1.0`. This is evidence for that
  scenario only, not bank admission. Calibrated 256/448 p95 costs are
  `223,001/390,083 s` (about `61.9/108.4 h`), so both operational time gates
  fail and no bank-scale profile is authorized. The clean-worktree receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_cost_confirmation_20260727_v1/validation_cost_confirmation.json`
  at SHA-256
  `f4bc393a7eabcdc058eb5f4de69281c5e1bed9feef275f9f75833f3f3c4aaae7`.
  S1 must optimize the same exact path and re-profile; it may not weaken or
  approximate the validator.
- [x] The first exact-path optimization probe removed 24 source-level
  `wrap_state` calls per candidate. The focused parity test in commit
  `7ff0cca1` preserved old/new outputs, but the cost probe measured no
  meaningful latency benefit: warmed four-row batch p50 changed from
  `0.072704` to `0.072606 s` (`0.9986x`), p95 changed from `0.073481` to
  `0.073782 s`, and projected 256 p95 changed from `216,001` to `218,679 s`.
  This null is consistent with compiler elimination or amortization of the
  unused derived outputs, but does not distinguish that explanation from
  remaining transition cost or timing noise. The patch is therefore rejected
  and reverted under the simplicity rule; no v2 full confirmation is warranted.
  The non-admission null receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_cost_probe_20260727_v2_rawwrap/validation_cost_probe.json`
  at SHA-256
  `174e604c4c27bda0c73a3e2a9457b1060472e1e1f26139e80aebc9d9ed624f05`.
  The next bounded treatment is a fixed-row 4/8/16 service-batch sweep with
  exact output parity and honest padded counters.
- [x] The batch-size treatment was frozen before execution. It reuses the pinned
  v1 probe identity, four-map source group, explicit initial state, protocol,
  inputs, CPU host, and unchanged exact service kernel. It builds the graph and
  prefilter once, hashes the first 16 accepted timing rows and the first 18
  accepted parity rows, and compares batch sizes `4`, `8`, and `16`. Every arm
  receives one untimed complete 16-row warmup followed by 12 equal-logical-work
  timed repeats; the 18-row pass forces tail padding for all three sizes.
  Dtype/shape/content-sensitive concatenated-output hashes must match batch 4
  exactly. A closing batch-4 control brackets the sweep and its complete-16-row
  p50 must stay within 5% of the opening control. Timing counts every padded
  launch; memory is cumulative for the shared process, not attributed to an
  arm; and a persistent compilation cache fails closed. The smallest eligible
  arm with minimum projected 256-scenario p95 is selected, but a change from 4
  requires that p95 to be strictly below the smaller opening/closing batch-4
  projected p50 and all memory/drift gates to pass. The sweep never calls the
  complete validator and never emits admission. A selected change authorizes
  only a fresh canonical cost probe and, if its first gate passes, one complete
  confirmation; it does not authorize the 256-scenario profile.
  The clean-worktree sweep at commit `af1bf5c4` passed exact parity for every
  arm with common output SHA-256
  `fa8dd4f8d579cd1c08ffd64876117f6be3ab41b6164a9220eacb37f4f64e1106`;
  all timing and cumulative-memory gates passed, and opening/closing batch-4
  complete-16-row p50 drift was only `2.27%`. Batch 4 opening/closing p50 was
  `0.3137/0.3067 s`, while batch 8 and 16 were slower at `0.3711/0.3389 s`.
  Their 256-scenario p95 projections were `75.5/75.1 h`, versus `65.4-65.5 h`
  for batch 4 and a strict faster-control p50 threshold of `62.8 h`. The frozen
  rule therefore retains batch 4 and authorizes neither a fresh probe/full
  confirmation nor the 256-scenario profile. The non-admission receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_batch_sweep_20260727_v1/direct_service_batch_size_sweep.json`
  at SHA-256
  `727c358357a026ff75b9e8340d6cd117efff6c782d455eba82630fe8e9312db9`.
- [x] The bounded exact-path candidate avoided building and dilating the
  truck-transfer cone when the frozen scenario has no other active truck.
  Commit `9f7b95fa` implements the empty-candidate early exit and preserves
  eager/JIT no-truck behavior plus the active-truck transfer path. It does not
  alter dump physics or the direct-service algorithm. The controlled
  cost-only probe reused the same pinned B0a identity/source group/state,
  `39,606` candidate rows, CPU host, and batch size. Warm batch p50 worsened
  from `0.072704` to `0.082937 s` (`+14.1%`) and p95 from `0.073481` to
  `0.090367 s` (`+23.0%`), failing both preregistered `>=5%` improvement
  gates. Commit `90c83808` reverts the candidate under the simplicity rule.
  No confirmation, 256-scenario profile, Static admission, or PPO is
  authorized. The non-admission receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_cost_probe_20260727_v3_no_truck_guard/validation_cost_probe.json`
  at SHA-256
  `cd5e60f9cfad0cb9e58a5112f9f87c9aa6701440b2fbc41e42bbe8f9ffdd810a`.
- [x] The execution-device parity subgate passed for the unchanged exact
  service kernel in tool commit `1856f92a`. The clean-worktree,
  single-RTX-4090 run left
  `JAX_PLATFORMS` unset and replayed the pinned first `18` accepted rows in
  five batch-4 launches (`20` padded rows). Candidate SHA-256
  `b19744abe759e0d229cc6a6d6095dd39beef3495311f10bb4ef7c2476438dd56`
  and the three `int32` output leaves matched the CPU dtype/shape/leaf/content
  SHA-256
  `fa8dd4f8d579cd1c08ffd64876117f6be3ab41b6164a9220eacb37f4f64e1106`
  exactly. Diagnostic process-lifetime peaks were `0.8653%` of normalized GPU
  capacity and `5.0526%` of host memory, both below `80%`; these are parity
  diagnostics, not cost evidence. The first interactive launch was externally
  terminated with exit `143` before writing a receipt; the successful retry
  ran under a user service. A Newton GPU benchmark began after the successful
  parity launch, so the exact content result remains valid, but no timing may
  be inferred from it. The non-admission receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_gpu_parity_20260727_v1/direct_service_gpu_parity.json`
  at SHA-256
  `6eee6e354c77924867a43f5311cabd029cc50e736f2ebd3360e25a0aea9baef1`.
- [x] The single authorized unchanged GPU batch-4 non-admission cost probe used
  the same identity, source group, inputs, protocol, stable initial state, and
  twelve core validator dependency hashes as the pinned CPU probe. Its timing and
  memory gates passed: one-scenario p95 was `315.702 s`, sequential `256/448`
  p95 was `25,179.938/43,901.244 s` (`6.99/12.19 h`), and normalized
  GPU/host peaks were `0.8653%/4.5182%`; the host normalization uses the
  physical-memory denominator recorded by the same-host parity receipt.
  However, the complete graph/prefilter population was not device-identical:
  CPU/GPU enumerated `31,366/31,418` reachable poses, `376,392/377,016`
  pose/cabin rows, and `39,606/39,520` exact-service candidates. Padding cannot
  explain the difference because padded rows are introduced only after graph
  expansion and acceptance. The first `18` rows therefore under-specified
  parity. Reject the pure-GPU exact validator path and do not run the R-32
  confirmation, bank profile, Static admission, or PPO from this result. The
  non-admission receipt is
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_gpu_cost_probe_20260727_v1/validation_cost_probe.json`
  at SHA-256
  `1250bdf0a723f5d739fce1dd6ccf90c9ea4eece5ece580da08fd3fd18e2971e1`.
  Any next validator treatment must preserve the CPU candidate population or
  demonstrate full-population parity before timing.
- [x] Test one bounded hybrid exact-validator treatment. Keep graph traversal
  and the dig prefilter on the canonical CPU, hash the resulting ordered
  unpadded `int32` candidate rows, transfer those rows unchanged, and run only
  `_service_batch` on the GPU. Before timing, require one full-population replay
  on the confirmed scenario to match the CPU candidate hash, every population
  counter, ordered unpadded output hashes, and final result from confirmation
  receipt
  `f4bc393a7eabcdc058eb5f4de69281c5e1bed9feef275f9f75833f3f3c4aaae7`.
  The receipt must explicitly hash `terra/benchmark_protocol.py` in addition
  to the existing dependency bundle. Only exact full-population parity
  authorizes profiling the hybrid path. If parity fails, retain the CPU exact
  path and evaluate process-level CPU scenario parallelism; do not change
  geometry, weaken the gate, run bank admission, or launch PPO.
  The clean `d66969fc` run reached the final comparator after both complete
  replays. CPU outcome, all population counters, candidate order, dispatch,
  and round-trip matched; ordered GPU service outputs and the hybrid final
  outcome did not. It exited `1` after `1,328 s` wall with no OOM. The
  success-only harness emitted no validator receipt, so the timestamped
  journal-derived failure record explicitly lists the evidence that was not
  persisted. Its path is
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_hybrid_parity_20260727_v1/hybrid_parity_failure.json`,
  SHA-256
  `49638f9b48fdc479cba5a01554c0afdbd71c6429886bacc33c877d883b1bc110`.
  Reject hybrid timing and retain exact CPU execution; only external
  scenario-level CPU process parallelism is eligible next.
- [ ] Test exactly one CPU process-scaling treatment before the 256-identity
  profile. Launch one fresh cohort of four long-lived spawned workers on fixed
  disjoint CPU affinity sets; each worker rematerializes the pinned confirmed
  state and calls the unchanged exact CPU entrypoint twice. Require all eight
  outcomes/counters to equal confirmation `f4bc393a...aae7`, four distinct
  successful PIDs, CPU-only backend, zero swap/OOM, complete code/input/state/
  protocol receipts, and conservative aggregate peak memory at most `80%`.
  With per-worker cold/warm totals `C_i/W_i`, freeze
  `P_N = max_i(C_i + (ceil(N/4)-1)*W_i)` and require
  `P_256 <= 86,400 s`, `P_448 <= 172,800 s`, and every call at most
  `3,600 s`. Do not sweep worker counts or retry a failed cohort. A pass
  authorizes only one four-worker 256-identity exact profile, not Static,
  admission, or PPO.
  The first service bootstrap is execution-null: all four workers failed
  module resolution before readiness, the start barrier, state materialization,
  or any exact call. Preserve its receipt at
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_direct_service_cpu_process_probe_20260727_v1/cpu_process_probe.json`
  (SHA-256
  `e09bb9719039754a64e8ca50286d2b1cbfd24d55334fe38c3f2f395bb1d8ce21`)
  and exclude its `0.417 s` coordinator lifetime
  from every timing or resource projection. This explicitly overrides only
  that receipt's mechanical `authorizes_retry: false`: one corrected v2 may
  launch workers as repository modules from the already pinned clean worktree,
  after asserting the imported Terra/profile/confirmation files originate
  there. If any corrected worker reaches readiness, no further cohort rerun is
  allowed.

The first recovery dense reward, named `corrected_dense_v1`, is the current
dense reward with one exact completion contract, contained mass-conserving
dumping, and no potential-based action veto. A candidate `transport_potential`
dense arm is only a recorded design question for now: if authorized, it must
replace rather than double-count the current dump-time relocation term and
must account for both off-zone soil and carried load. Its equation and distance
metric remain undecided.

## 2. Critical assessment of the Oracle review

The Oracle review is useful because it separates confirmed code defects from
hypotheses and does not respond to poor learning with an architecture sweep.
Its central recommendation—prove semantics and dynamic feasibility first—is
sound. It needs the following corrections and additions.

| Oracle claim | Assessment | Consequence |
|---|---|---|
| Termination and terminal completion use different dump masks. | Confirmed. Termination accepts a one-cell dilation while completion and reward count exact target cells. | This is a correctness blocker, but the repair must first decide which mask is legally authoritative. Blindly widening reward would hide the map's visible dump constraint. |
| The `3/3` controller is not a competence curriculum. | Confirmed and strengthened by the final staged results. It pools outcomes from random identities, families, and cells. | Retire it as the candidate curriculum. Replace it with global checkpoint-bounded stages and fixed promotion-bank gates. |
| M0-M2 static validation does not prove dynamic feasibility. | Confirmed. | Run fixed-identity learning probes before another family or generalist campaign. |
| Flat suffered catastrophic forgetting. | Too strong. It showed held-out regression without ever mastering M0, under a stationary mixture. | Call it retention/generalization regression. Separate memorization, mixture interference, and PPO drift. |
| Positive soil height clipping may alias important states. | The clipping is confirmed; its causal importance is not. Local features may recover some nearby volume. | Require a paired-state alias test before changing observation preprocessing. |
| Random initial `env_steps` violates the full-reset contract. | Confirmed, but it affects only the first episode of each training process and cannot explain late regression alone. | Remove it for future full-reset experiments; do not claim it explains the failed screen. |
| No architecture experiment should be next. | Correct. Existing old-map results are saturated and do not rank architectures on the new distribution. | Retain `_se`; xattn is the first conditional architecture ablation only after semantic, feasibility, and retention gates. |
| Progressive reward v2 is ready after dense qualification. | Not yet. Its current Stage-1 legacy-parity requirement conflicts with its corrected completion invariant. | Revise Stage 1 to use a newly frozen corrected dense contract before implementing Stage 2 or Stage 3. |
| The staged result was incomplete. | Stale at review time. The evaluator later completed. | The final result strengthens rejection: staged peaks at M0 13/64, M1 2/64, M2 0/64. |

The review also underweights two cheap, high-information diagnostics:

- evaluate exact training identities and source-disjoint development identities
  with the same checkpoint and action-selection mode; and
- compare deterministic argmax with repeated sampled-action evaluation on a
  small selected checkpoint set.

Both should precede new PPO training.

The current data volume is another material weakness: training has only eight
unique M0 maps and twelve unique M1/M2 maps per primary cell. Repeated slots
balance sampling but do not create procedural diversity. Larger source-disjoint
training banks become mandatory after single-map feasibility is established.

## 3. Experimental rules

1. Preserve the historical revisions and reward semantics for retrospective
   diagnosis. Do not mix a semantic fix into a continuation intended to explain
   the old run.
2. Use the corrected contract for every future-policy feasibility, specialist,
   curriculum, reward, reset, and architecture experiment.
3. Change one causal factor per comparison.
4. Use untouched 450-step full resets with `env_steps == 0` unless reset
   distribution is the named treatment.
5. Keep map, reward, reset, and architecture curricula separate.
6. Use fixed source-disjoint banks for decisions. Online success, return,
   entropy, and curriculum occupancy are diagnostics.
7. Treat single-map learning as a dynamic feasibility witness, never as a
   generalization claim.
8. Fail loudly on missing maps, metadata, distance fields, or evaluator
   integrity fields. Do not add fallback loaders or a generic curriculum
   framework.
9. Do not use the development or sealed bank to drive automatic promotion.
   Promotion uses a separate frozen source-disjoint gate bank.
10. Every training task below requires its own update-1 finite smoke and exact
    run receipt before a production launch.
11. Treat
    `sum(world action-map soil) + sum(all active carried loads)` relative to
    the reset state as a transition invariant. A map, dump, soil-relaxation, or
    partial-reset path that clips, deletes, creates, or overflows soil is
    invalid.
12. Use the small base architecture for recovery feasibility and teacher
    training. Do not mix an architecture comparison into those runs.
13. Short budgets are decision milestones, not evidence that learning has
    saturated. On a deterministic source-disjoint bank, "slight improvement"
    means either one additional successful identity in the family or current
    worst cell, or at least `0.01` absolute improvement in median terminal task
    completion, relative to the previous best scheduled evaluation. Reward,
    loss, or online success alone cannot trigger continuation.
14. At a milestone, continue the exact model, optimizer, RNG, environment,
    action-history, and schedule lineage to the next
    `500 -> 1,000 -> 2,000 -> 5,000` milestone whenever slight improvement
    exists and integrity remains clean. A five-evaluation fixed-bank plateau
    with no such improvement is the stop rule. The current checkpoint-v2
    resume path restores model, optimizer, update, and schedule position but
    explicitly restarts environment, RNG, and action history; it therefore
    does **not** satisfy this rule. Until an exact checkpoint contract exists,
    an extended confirmation must be launched as one fresh continuous run at
    the full declared milestone and reported as a repeat, not as an exact
    continuation.
15. A recipe is considered qualified for long training only after its
    source-disjoint family/cell gate passes twice and no semantic, integrity,
    or causal blocker remains. Family specialists qualify independently; one
    family never waits for the other before receiving its own long run.
    Slight improvement without two gate passes still earns the next declared
    bounded milestone under rules 13-14, but is not yet production
    qualification. Qualified continuations use Euler
    `gpuhe.120h`, request `5-00:00:00`, run for at least 20,000 updates, save
    fixed-bank checkpoints at a declared cadence, and receive more compute
    while rule 13 still shows improvement. Until exact checkpointing is
    implemented, the first selected production treatment is one continuous
    20,000-update run; a higher milestone must be a fresh continuous run at
    the full cumulative budget rather than a checkpoint-v2 pseudo-resume.
    With an exact checkpoint contract, extend in 20,000-update chunks. Long
    training never changes map, reward, reset, PPO, or architecture treatment
    in place.
16. A qualified long run uses a separate production launcher, never the
    bounded B0 screen launcher. Before submission, prove the exact qualified
    recipe and train/promotion/development bank hashes, scratch space and inode
    headroom, four RTX 4090s, CUDA/cuDNN/NCCL preflight, `gpuhe.120h`,
    `5-00:00:00`, `2,621,440,000` transitions for 20,000 updates, update-1
    finite smoke, numbered-checkpoint cadence, dependent terminal verifier,
    and fixed-bank evaluator. Reject `--resume_from` until a checkpoint
    contract serializes the full runner state. Seal selected final artifacts
    by checksum-copying them to a verified persistent destination with enough
    capacity; a scratch-only checkpoint is not a durable result.

The production launcher is a narrow qualified-recipe path, not another generic
curriculum framework. Implement it only against the first concrete qualified
F1 recipe, reusing the proven B0 hashing, GPU, smoke, aggregate, checkpoint,
and evaluator helpers without reusing the bounded B0 sbatch itself. Its frozen
acceptance contract is:

- exactly 200 numbered checkpoints at updates `100, 200, ..., 20,000`, with
  `FINAL` model and optimizer state exactly equal to update 20,000;
- deterministic promotion-bank evaluation at all 200 checkpoints and
  development evaluation at `500, 1,000, ..., 20,000` (40 checkpoints);
- streaming checkpoint verification/evaluation rather than simultaneously
  loading all 200 policies;
- 20,000 ordered per-update aggregate receipts with hard-zero mass,
  target/obstacle mutation, nonfinite, and reward-residual violations;
- a dependent finalizer that verifies the terminal contract, evaluates the
  fixed banks, and checksum-copies source, bank, receipts, checkpoints,
  aggregates, evaluations, and logs to a verified persistent destination such
  as `/cluster/work/rsl/lterenzi`; and
- at launch time, at least 16 GiB and 50,000 inodes of scratch headroom for
  that run. Current estimates are roughly 2.9-4 GiB for 200 base checkpoints
  plus receipts before the durable archive, but live space must be refreshed.

The final five scheduled promotion evaluations apply rule 13. Until runner
state is exact, an improving 20,000-update run authorizes a fresh continuous
40,000-update repeat, not `--resume_from`; after exact runner-state
checkpointing exists, it may instead extend in genuine 20,000-update chunks.

In this document, **manifest-sealed** means that the declared source, bank,
configuration, and receipt bytes are content-addressed and must still match
their frozen SHA-256 manifests when consumed. It does not imply filesystem
write protection unless a separate permission receipt says so. Unmanifested
runtime files such as `__pycache__` do not enter the causal package, but any
drift in a manifest-listed file invalidates the evidence. Historical uses of
“immutable root” should be read under this exact contract.

## 4. Dependency graph

```text
D0 frozen first-screen receipt
 |
 +--> D1 historical semantic attribution -----> optional R0 historical fork
 |
 `--> D2 train/dev and action-mode audit ------^

C0 exact visible dump mask (ratified)
 |
 +--> C1 one completion contract
 |
 `--> C1a contained mass-conserving dump transition
          |
          +--> C2 full-reset horizon
          +--> C3 exact loader
          +--> C1b exact footprint rasterization
          `--> C4 minimal fixed evaluator
                    |
                    +--> C5 auditable training aggregates
                    |
                    `--> F0 two scratch fixed-identity probes
                            | fail
                            `--> O0/transition/reward diagnosis
                                      |
                                     F0R one-factor trench repair
                                      |
                                     B0 orthogonal feasibility cells
                               |
                              F1 two scratch family specialists
                               |
                              G0 scratch easy small generalist
                               |
                              S0 grow and qualify medium student
                               |
                              K0 global checkpoint-gated map ladder
                             /                 \
                  reward curriculum       partial resets

Architecture work remains conditional on a representation-specific failure.
```

`R0` is the sole historical-revision experiment. `F0` and every other
future-policy task use the corrected contract. D1 and D2 improve historical
attribution but do not block the already-ratified future semantic correction.
C5 and C1b are now complete and are required launch gates for every new PPO
run.

Task index:

| ID | Priority | Cost class | Depends on | State |
|---|---|---|---|---|
| D0 | P0 | documentation | completed evaluators | [x] complete |
| D1 | P0 | evaluation only | D0 | [x] diagnosis complete; historical mass-integrity gate failed |
| D2 | P0 | evaluation only | D0 | [x] 20 deterministic plus 16 sampled records adjudicated |
| C0 | P0 | decision | design review | [x] complete |
| C1 | P0 | Terra code/tests | C0 | [x] complete |
| C1a | P0 | Terra transition/tests | C0 | [x] complete |
| C1b | P0 | Terra geometry/tests | C1a | [x] complete |
| C2 | P0 | baselines code/test | C1, C1a | [x] complete |
| C3 | P0 | Terra loader/tests | C1, C1a | [x] complete |
| C4 | P0 | evaluator/tests | C1-C3, C1a | [x] complete |
| C5 | P0 | training receipts/tests | C1-C4 | [x] complete |
| O0 | P1 | conditional deterministic tests | failed F0 or direct alias evidence | [x] alias test not authorized: trajectory evidence implicates action/reward attractors |
| F0 | P0 | two scratch bounded PPO probes | C0-C5, C1a, C1b | [x] foundation passed; trench failed |
| F0R | P0 | one scratch trench reward repair | failed trench F0, diagnosis | [x] passed with terminal retention |
| R0 | P1 | two 500-update historical forks | D1, D2, F0 | [x] not authorized: shared train-and-development drift rejected |
| B0 | P1 | paired generation, five bounded panel probes, bank expansion | foundation F0, trench F0R | [ ] B0a passed; B0b active |
| F1 | P1 | two scratch family specialists | B0, foundation F0, trench F0R, C5 | [ ] blocked |
| G0 | P1 | one scratch small easy generalist | F1 | [ ] blocked |
| S0 | P1 | one grown medium qualification | G0 | [ ] blocked |
| K0 | P2 | global staged map campaign | S0 | [ ] blocked |
| A0 | P3 | one conditional architecture A/B | representation-specific evidence | [ ] blocked |
| W0 | P2 | corrected reward specification | C1, C1a | [ ] blocked |
| W0a | P2 | conditional dense transport A/B | F0 plus transport-specific evidence | [ ] blocked |
| W1-W2 | P2 | dense-to-terminal reward A/B/C | S0 | [ ] blocked |
| PR0 | P3 | reset A/B | selected K0 map sampler | [ ] blocked |

## 5. Phase D — close and diagnose the completed screen

### D0 — Freeze the first-screen receipt

Status: complete.

Record:

- Terra revision `d37e780480c0fae64a4b9e4ba6638b4499748761`;
- terra-baselines revision `2722d832c8381a68d594d8bf8298ba3aec7f4c6a`;
- flat job `8398905`, W&B `mwn0d0cr`;
- staged job `8398906`, W&B `kh3pf9tw`;
- evaluator jobs `8398907` and `8398908`;
- evaluator JSON hashes; and
- the complete family/cell checkpoint cadence.

Acceptance:

- both arms are recorded as rejected;
- neither is called mastered or selected;
- the sealed bank remains unopened; and
- no second paired seed is authorized.

### D1 — Audit the dump-mask semantic mismatch

Execution receipt, retry submitted 2026-07-26:

- observer-only implementation:
  terra-baselines `d049107` plus the terminal-threshold default fix
  `1aeb1a6` (`audit_historical_curriculum.py`);
- frozen source copy:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/historical_audit/source`;
- historical Terra and baseline revisions remain
  `d37e780480c0fae64a4b9e4ba6638b4499748761` and
  `2722d832c8381a68d594d8bf8298ba3aec7f4c6a`;
- initial preflight `8623160` failed before producing evidence because the
  observer directly accessed an optional historical terminal-threshold field;
  its dependent jobs `8623162` and `8623163` were cancelled automatically;
- first replacement `8624492` completed the audit but the shell gate rejected
  a maximum `1.43e-6` float32 reconstruction difference against a `1e-6`
  threshold; that JSON is preserved under `failed_attempts/`;
- the explicit, receipt-recorded float32 tolerance is now `1e-5`
  (`7b5d52d`), with regression coverage;
- replacement preflight `8626340`, completed in `00:09:09` with exit code
  `0:0`;
- replacement full deterministic job `8626341` produced all 20 declared records
  and exact reset receipts, then exited `1:0` because its final shell gate found
  historical dump-observer transition mismatches;
- deterministic JSON SHA-256:
  `22abef7ca13006d31abf2bba2d268581ea5298532b5fe1fd8ece0c976e69df1b`;
- 15 M1/M2 timeout episodes have both a nonzero mass residual and at least one
  dump attempt for which the observer did not reproduce the old transition;
  their dump-veto and boundary-flow diagnostics are not admissible yet;
- bounded diagnostic implementation `04edebf` records which branch mismatched
  plus the first-step load and terrain deltas;
- diagnostic job `8632822` targets only `flat_u1000` on development M1
  deterministically, at most 28,800 transitions, under immutable diagnostic
  root
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/historical_audit/diagnostics/observer_mismatch_v1`;
- diagnostic launch-receipt SHA-256:
  `ffbeca6347e73bcfd44599521623599a1708cd91e79c3b702dcd231b9b315f88`;
- diagnostic job `8632822` completed in `00:10:22` with exit code `0:0`;
- diagnostic JSON SHA-256:
  `ca61654b1e1acdefdc61c7ccc888e6d854b03669e83fc81439ac4a21efb4e526`;
- all three mismatches in the selected `flat_u1000` development-M1 record were
  non-veto executed dumps, not load-state or veto-branch classification errors:
  carried loads 43, 37, and 33 became zero while actual world-soil deltas were
  only 42, 36, and 32;
- the corresponding independently reconstructed candidate deltas conserved all
  43, 37, and 33 units and differed from the actual maps by L1 values 3, 1,
  and 1. The historical transition therefore deleted one soil unit in each
  selected episode;
- every one of the original 15 transition-observer mismatches co-occurs with a
  nonzero historical mass residual. The bounded diagnostic identifies the
  selected three but does not assume that all remaining twelve share the exact
  same internal path;
- observer-derived potential-veto and boundary-flow counts are inadmissible for
  those 15 episodes. Actual terminal state, success, exact/buffer completion,
  terminal reward, and independently measured mass residual remain observable;
  and
- the deterministic command hard-limits D1 attribution to the three declared
  checkpoints over development M0-M2: exactly 259,200 maximum transitions.

D1 is a completed diagnosis but not a passed clean causal audit: the historical
transition itself fails mass integrity. Do not erase or waive that gate.
Materiality conclusions below use the fully clean M0 records and a sensitivity
view that excludes every integrity-failing M1/M2 episode.

Hypothesis:

> Buffer-only legal terminations are receiving a different completion and
> terminal reward from equivalent exact-zone terminations.

Implementation:

- add read-only counterfactual metrics to fixed evaluation;
- do not change transitions, actions, task termination, or training;
- evaluate flat update 1,000, flat update 4,000, and staged update 4,000;
- use development M0-M2 at horizon 450; and
- record exact-target and current accepted-buffer completion on every terminal
  state and top-quartile timeout.

Required output by family and primary cell:

- `task_done && exact_completion < 1`;
- exact-mask completion;
- accepted-mask completion;
- current versus counterfactual terminal reward;
- positive soil volume in buffer-only cells;
- dump attempts rejected solely because predicted relocation potential
  increased;
- positive-soil delta moved across the visible target boundary by local soil
  relaxation;
- mass residual; and
- timeout completion delta.

Decision:

- any `task_done && completion < 1` confirms the contract violation;
- call it a material contributor only if at least 10% of successes or at least
  10% of top-quartile timeouts change completion by `>= 0.05` or terminal
  reward by `>= 0.1 * terminal_reward`;
- report the veto and boundary-crossing rates separately rather than folding
  them into the completion threshold; and
- otherwise fix it as correctness debt without claiming it caused the broad
  regression.

Budget: at most 259,200 evaluation transitions and no gradients.

Adjudication:

- the exact-versus-buffer completion mismatch is a material contributor under
  the preregistered threshold, but it is not established as the sole cause of
  poor generalization;
- on clean development M0, `flat_u1000` changes 11/24 successes overall:
  1/11 foundation successes (9.09%, below the family threshold) and 10/13
  trench successes (76.9%);
- on clean development M0, `flat_u4000` changes 2/11 successes (18.2%);
- on clean development M0, `staged_u4000` changes 1/11 successes and 3/14
  top-quartile timeouts, so the timeout criterion is material;
- after excluding all integrity-failing rows, the aggregate materiality
  decision remains positive for each declared M1 and M2
  checkpoint/dataset record;
- potential-veto attempts and boundary relaxation are present in clean rows,
  but their rates must be reported separately and cannot be extrapolated from
  the 15 integrity-failing episodes; and
- the corrected future contract—exact visible mask everywhere, contained
  mass-conserving dumping, and no relocation-potential action veto—directly
  removes all three confounds. No historical R0 continuation may be used as
  evidence for that corrected contract.

### D2 — Separate memorization, policy mode, and held-out regression

Execution receipt, retry submitted 2026-07-26:

- deterministic train/development audit job `8626341` saved all 20 records but
  exposed the D1 observer-integrity issue described above;
- sampled M0 job `8626343` produced the first three complete records, then was
  intentionally cancelled after `01:38:03` because its serial execution rate
  could not finish the exact grid inside the four-hour allocation;
- the preserved `flat_u1000` seeds `2026072500` through `2026072502` JSON has
  SHA-256
  `a63e7b1d44aebda1d3774c11568b9bb6aaa0c929463bf14e7a911a2a08b3158a`;
- execution-only sharding at terra-baselines `b34c122` leaves checkpoints,
  maps, seeds, horizon, action sampling, and historical source unchanged;
- immutable shard root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/historical_audit/sampled_shards_v1`;
- shard launch-receipt SHA-256:
  `9019d01b7515554df086ecc18ca0fafee8cada5c40aea94287eb8512384e3cff`;
- submitted-jobs receipt SHA-256:
  `db112eb390acf8722cc0d5144196a10ff70577854f93916179a4d8a111fad50e`;
- jobs `8633209`, `8633211`, and `8633220` cover, respectively,
  `flat_u1000` seeds 3-7 and `flat_u4000` seeds 0-3 and 4-7, with no overlap;
- those jobs completed with exit code `0:0` in `00:17:35`, `00:33:57`, and
  `00:34:00`;
- shard JSON SHA-256 values, in input order, are
  `a63e7b1d44aebda1d3774c11568b9bb6aaa0c929463bf14e7a911a2a08b3158a`,
  `ac0b6f799172f21bbb063df501eb8cee205f8492b04d96fa4f860d732408326d`,
  `b92bea1538a6939dd3271a9652f94be2f8b230b31eb4a89ce337aa96becdacb6`,
  and
  `c0cbd9ca9b3fc0226a3013d4db5978e40c25d5abc23a14a72decf2658a6063d1`;
- the exact 16-record merge passed uniqueness and static-contract checks and
  has SHA-256
  `3ad4805fb64b4bb8b08ed25b32c2903233e2ff551cad87c982a3b37cd9759a74`;
- declared sampled seeds `2026072500` through `2026072507`;
- exact training-identity view:
  `train/local_M2_terminal`, whose 256 slots must verify as 256 unique source
  IDs and 256 unique map IDs before evaluation; and
- checkpoint labels `e8_u20000`, `flat_u1000`, `flat_u4000`,
  `staged_u1000`, and `staged_u4000`, each guarded by its frozen SHA-256.

D2 execution and diagnosis are complete, but the historical integrity gate did
not pass. The deterministic records contain 48/2,240 episodes with nonzero
historical mass residual (maximum 3), and sampled records contain 3/1,024
episodes with one-unit residuals. None of those 51 episodes succeeded; all
reset hashes, target/obstacle immutability, finite-state, and terminal-reward
reconstruction checks passed. Results below retain the failures explicitly and
include the clean-row sensitivity where relevant.

Hypotheses:

1. online success is high because the policy fits repeated training identities;
2. deterministic argmax under-reports a diffuse but useful sampled policy; or
3. both train and development behavior regress because optimization or
   heterogeneous exposure moves the policy away from M0.

Implementation:

- build an exact manifest over unique training identities, excluding repeated
  slots;
- evaluate E8 zero-shot, flat updates 1,000 and 4,000, and staged updates 1,000
  and 4,000;
- run deterministic evaluation on train identities and development M0-M2;
- for flat updates 1,000 and 4,000 only, repeat sampled-action evaluation with
  eight declared seeds on M0;
- report family and primary-cell results; and
- include policy entropy, action-logit margin, and deterministic-versus-sampled
  action disagreement.

Interpretation:

- support memorization when train success is at least 60% and exceeds
  source-disjoint development by at least 20 percentage points in the same
  family;
- support shared optimization drift when train and development success both
  fall by at least 10 percentage points from update 1,000 to update 4,000;
- support diffuse-policy sensitivity when mean sampled M0 success exceeds
  deterministic M0 success by at least 10 percentage points and at least six
  of eight sampled seeds improve;
- none of these outcomes alone establishes architecture insufficiency.

Acceptance:

- train and development use identical horizon, reset, and action-selection
  semantics;
- repeated training slots are not double-counted; and
- the report states which hypotheses remain viable.

Adjudication:

- the exact training view passed with 256 unique source IDs and 256 unique map
  IDs, so repeated slots were not double-counted;
- the memorization criterion is supported for the trench family at
  `flat_u4000`: training reaches 82/128 (64.1%) while source-disjoint
  development M0 reaches only 3/32 (9.38%), a 54.7-point gap even against the
  easiest development level. Foundation and overall training do not reach the
  60% threshold;
- shared train-and-development drift is rejected. From `flat_u1000` to
  `flat_u4000`, unique-training success rises from 54/256 to 122/256 while
  development M0 falls from 24/64 to 11/64. Both families improve on training
  identities, so the preregistered joint-decline condition is false;
- diffuse-policy sensitivity is supported at both checkpoints. Update 1,000
  sampled counts are 31, 29, 36, 34, 30, 35, 34, and 29 (mean 32.25/64,
  +12.9 points over deterministic 24/64); update 4,000 counts are 21, 19, 20,
  24, 20, 20, 22, and 19 (mean 20.625/64, +15.0 points over deterministic
  11/64). All 8/8 seeds improve at each checkpoint;
- sampled action disagreement averages 35.1% at update 1,000 and 20.0% at
  update 4,000, consistent with useful probability mass outside the argmax;
- the three sampled mass-integrity failures all occur at update 1,000, are
  unsuccessful episodes, and do not change either sampled-action decision
  under clean-row sensitivity; and
- the viable explanation for the old screen is family-specific identity fit
  plus held-out generalization regression, compounded by a material task
  contract mismatch and deterministic action-mode sensitivity. It is not
  evidence for architecture insufficiency or for shared optimization drift.
  R0 is therefore not authorized.

## 6. Phase C — establish one corrected future-policy contract

### C0 — Legal dump-mask decision

Status: complete.

Ratified contract:

> The explicit map dump mask is the legal dump region. Any tolerance region
> must be materialized by the generator, visible in the map/gallery, included
> in capacity calculations, and stored in the manifest. There is no hidden
> one-cell legal buffer.

For the current representation, the direct definition is
`accepted_dump_mask = (target_map > 0) & ~obstacle_mask`, with generation
rejecting any target/obstacle overlap.

This recommendation preserves the meaning of arbitrary dump constraints and
prevents the environment from silently accepting soil outside the reviewed
zone. An off-zone dump may remain a physically executable mistake; it is never
part of the accepted mask or successful completion.

Deliverable:

- one named `accepted_dump_mask` definition;
- a short design decision in `TRAINING_DESIGN.md`; and
- no second termination-specific or reward-specific dump mask.

Decision receipt: ratified with Lorenzo on 2026-07-25. No further mask choice
is required before implementation.

### C1 — Unify termination, completion, reward, and evaluation

Status: complete.

Verified implementation receipt, 2026-07-25:

- Terra exposes the named `exact_visible_dump_v1` contract and computes dig,
  exact-dump purity/volume, unloaded, task-present, dump-mask-integrity, and
  applicable edge components once;
- termination, terminal reward, reward components, and the compatibility
  completion accessor use the same minimum-reduced absolute completion;
- exact-zone, former-buffer, off-zone, relocation-only, combined, partial,
  loaded, empty-task, obstacle-overlap, edge, terminal-reward, eager, `jit`,
  and `vmap` tests pass; and
- the C4 evaluator now records the same named contract, hard-fails any
  `task_done <=> absolute_completion == 1` disagreement, and labels the
  imported environment as either corrected or legacy; and
- all 111 baseline tests and the real corrected-environment C4 smoke pass.

Implement the smallest pure task-completion path needed by the current dense
experiment:

- compute dig, dump, unloaded, and applicable edge requirements once;
- reduce active requirements with a minimum so one completed component cannot
  hide an unfinished terminal prerequisite;
- use the same accepted dump mask for termination, completion, reward logging,
  and evaluation;
- require `task_done <=> absolute_completion == 1` within tolerance;
- create a new named corrected dense contract;
- do not claim numerical parity with the inconsistent legacy terminal
  calculation; and
- do not implement the full progressive reward framework in this task.

Primary files:

- `terra/state.py`;
- focused Terra completion tests; and
- the fixed evaluator's completion reporting.

Required deterministic cases:

- exact-zone soil;
- soil in the former hidden buffer;
- soil outside the legal region;
- obstacle-overlapping buffer cells;
- relocation-only and combined dig-and-dump tasks;
- incomplete and complete digging;
- loaded and unloaded agents;
- foundation edge incomplete/complete; and
- mass-conserving partial states.

Acceptance:

- every success has completion one;
- every completion-one state succeeds;
- no terminal reward path recomputes completion differently;
- eager, `jit`, and `vmap` cases agree; and
- legacy checkpoints remain evaluable under a clearly labeled legacy contract.

### C1a — Make dumping contained, mass-conserving, and non-greedy

Status: complete.

Verified implementation receipt, 2026-07-25:

- tracked-excavator dumps prefer physically reachable exact-mask cells and
  otherwise permit an entirely off-zone, recoverable mistake;
- correctly aimed and wrong dumps constrain local soil relaxation to opposite
  sides of the accepted-mask boundary;
- complete-load mass, containment, and `int8` representability are checked
  before changing world soil or carried load;
- reward-potential transition vetoes were removed from active dump and
  implicit reverse-dump paths;
- obstacles and non-dumpable cells are excluded per tile instead of vetoing an
  otherwise usable dump workspace; and
- 16 focused contract tests plus all 39 Terra tests passed,
  including repeated dump/re-lift, overflow rejection, potential increase,
  and eager/`jit`/`vmap` agreement.

Capacity-validator receipt, 2026-07-25:

- the loader computes accepted cells from the exact visible mask, rejects
  target/obstacle and target/non-dumpable overlap, checks the declared
  single-layer ratio, and verifies total and maximum-bucket `int8` headroom;
- focused insufficient-area and unplaceable-bucket fixtures pass;
- the old `terra_training_design_v1_20260724` bank was audited and has minima
  near 2.0x, so it is explicitly ineligible for the new 3x starter contract;
  and
- regenerated F0 receipts now prove 63.0x capacity and one-tile path distance
  for the all-around foundation, and 9.27x capacity with two-tile p95 path
  distance for the broad both-side trench. Both are obstacle-free, exact-mask
  datasets and load successfully through the strict C3 path; the frozen bank,
  gallery, and validation JSON are under
  `.artifacts/terra_curriculum_recovery_20260725/f0_starters_v1/`.

Scope the first implementation to the single tracked-excavator recovery path.
Do not build a configurable spill framework.

Transition rule:

1. Build the physically reachable dump workspace using the existing geometry,
   obstacle, traversability, and dumpability constraints.
2. If that workspace intersects `accepted_dump_mask`, interpret the action as
   a correctly aimed dump. Deposit the complete carried load only on the
   reachable accepted cells and restrict local soil relaxation to the accepted
   mask.
3. If the workspace contains no accepted cell but contains physically
   dumpable off-zone cells, permit the wrong dump and restrict its deposition
   and relaxation to off-zone valid cells. The resulting soil remains illegal
   until recovered.
4. Never move soil across the accepted-mask boundary during either path.
5. Never clip or delete an unplaceable remainder. Fail the dump without
   changing world soil or carried load if the complete load cannot be
   represented.

Remove relocation-potential comparisons from action validity and transition
acceptance. In particular, the active `_handle_dump` path must not return the
unchanged state merely because predicted relocation potential is higher. Any
equivalent veto in an action-availability or implicit reverse-dump path must
also be removed for the active recovery agent.

The map validator must compute capacity from the exact accepted mask. Starter
cells use at least `3x` reachable single-layer-equivalent capacity and must
also prove that all valid bucket loads stay within the action-map numeric
range under the contained pile rule.

Required deterministic cases:

- an interior legal dump;
- a legal dump whose unconstrained soil relaxation would cross the boundary;
- a workspace overlapping both legal and neutral cells;
- an entirely off-zone wrong dump;
- a legal region with insufficient representable capacity;
- a dump adjacent to an obstacle or non-dumpable tile;
- a dump that increases relocation potential but is physically valid; and
- repeated dig/lift/dump/re-lift sequences.

Acceptance:

- `sum(action_map) + sum(active loaded soil)` is exactly conserved on every
  successful transition and unchanged on every rejected transition;
- a legal dump creates no positive-soil delta outside
  `accepted_dump_mask`;
- an off-zone dump creates no positive-soil delta inside
  `accepted_dump_mask`;
- carried load decreases by exactly the soil added to the world;
- no integer overflow, wraparound, clipping, or silent remainder loss occurs;
- the former potential-increase veto has no effect on transition legality; and
- eager, `jit`, and `vmap` cases agree.

Physical boundary spill becomes a later named dynamics treatment only after
the contained contract passes family and retention gates.

### C1b — Preserve the exact excavator footprint

Status: complete.

Verified implementation receipt, 2026-07-26:

- the pre-existing polygon rasterizer sampled integer cell corners under
  strict half-plane tests, shrinking an axis-aligned `W x H` footprint to
  `(W-1) x (H-1)`, and its x/y grid construction was transposed;
- Terra commit `f3eeca6a` samples cell centers in `[x, y]` order and returns a
  `(map_height, map_width)` mask;
- a regression built from the production `get_agent_corners` path proves that
  an odd `5 x 3` excavator occupies exactly 15 correctly oriented cells;
- the focused footprint, dump-contract, and partial-loading set passes
  29 tests; and
- the full Terra suite passes 51 tests plus 6 subtests, with formatting,
  linting, and whitespace checks clean.

This issue was discovered while closing the training-integrity gates and was
fixed before any corrected-contract PPO production launch.

Acceptance:

- centered and boundary-touching footprints preserve their declared cells;
- non-square footprints are not transposed;
- odd production dimensions occupy exactly `width * height` cells at zero
  rotation; and
- dump and partial-loading transition tests remain green.

### C2 — Restore the full-reset horizon contract

Status: complete.

Verified implementation receipt, 2026-07-25:

- the initial full-task reset no longer randomizes `env_steps`;
- the training path asserts that every initial `env_steps` value is zero;
- the run metrics record the minimum and maximum configured effective
  horizon, while the direct fixed evaluator rejects any horizon other than
  450;
- no `randomize_initial_env_steps` reference remains; and
- all 42 focused training-utility tests passed on CPU, including zero and
  nonzero reset fixtures.

For all future full-task training and evaluation:

- remove `randomize_initial_env_steps` from the initial reset path;
- assert `env_steps == 0` after reset;
- keep the 450-step horizon identical across train and fixed evaluation; and
- treat randomized remaining horizon as a future named treatment, not a
  startup optimization.

This is a correctness cleanup, not the proposed explanation for late
regression.

Acceptance:

- a focused reset test observes zero elapsed steps;
- the training receipt logs the effective horizon; and
- no hidden countdown randomization remains in a full-reset preset.

### C3 — Make dataset loading exact and fail loud

Status: complete.

Verified implementation receipt, 2026-07-25:

- future multi-map loading requires the named
  `terra_exact_map_dataset_v1` contract before constructing JAX arrays;
- `DATASET_SIZE`, declared slot count, manifest rows, contiguous indices, and
  all target/action/occupancy/dumpability/distance/metadata sidecars must
  agree exactly;
- the contract records unique identities, explicit per-slot weight and
  identity multiplicity, shape, distance metric/normalization, exact dump
  contract, and an optional capacity floor;
- a hashed source registry is verified and rejects a source ID assigned to
  more than one split;
- missing sidecars, count/multiplicity mismatches, source overlap, invalid
  distance data, and a violated capacity floor all have deterministic failure
  fixtures; and
- all 44 Terra tests pass. The partial-reset generator uses one explicit
  legacy-contract test opt-out until its separate PR0 bank gate is active.

Change the current short-dataset warning into an error. Validate before JAX
compilation:

- exact expected map count;
- contiguous indices;
- target/action/occupancy/dumpability/distance files for every index;
- finite arrays and declared shapes;
- source-disjoint split IDs;
- manifest slot and unique-identity counts; and
- distance metric and normalization metadata.

Acceptance:

- a missing map or sidecar fails before environment construction;
- no zero or default distance map is substituted; and
- M1 slot multiplicity is explicit rather than silently weighted.

### C4 — Complete the fixed evaluator contract

Status: complete.

Verified implementation receipt, 2026-07-25:

- direct policy evaluation can preserve each first terminal state instead of
  auto-resetting it; inactive environments are frozen for the rest of the
  batch rollout;
- exact reset verifies target, initial action, occupancy, initial
  dumpability, reward distance, trench/foundation metadata, and zero elapsed
  steps, with aggregate layer hashes in the JSON receipt;
- deterministic versus sampled mode, manifest provenance and slot weights,
  verified reset slot, completion components, mass residual, no-effect action
  count, immutable-map mutation, non-finite state, and termination/slot
  disagreement are saved per map;
- any integrity failure blocks `mastery_gate.passed`, while legacy
  environments remain labeled and cannot earn corrected-contract mastery
  without the integrity fields;
- `aggregate_fixed_bank_history.py` requires two adjacent passing
  checkpoints and currently evaluates the historical five-percentage-point
  family retention rule;
- focused fixtures prove that one perfect-performing map with a mass error
  cannot pass and that one checkpoint cannot claim consecutive mastery;
- all 111 baseline tests pass; and
- a real one-step foundation rollout with a scratch
  `resnet_spatial_8x8` model reported supported integrity, slot 0, zero mass
  residual, no target/obstacle mutation, and no non-finite state.

Keep one direct fixed-bank evaluator and add:

- explicit deterministic or sampled mode in every receipt;
- map ID, source ID, family, stratum, primary cell, and slot weight;
- selected slot index verified from the exact reset key and preserved in the
  fixed evaluator's terminal accumulator so outcomes join to manifest
  provenance; C5 separately carries provenance through high-volume training;
- exact verification of target, initial action, occupancy, dumpability,
  distance, and relevant metadata at reset;
- mass residual;
- invalid or no-op action counts;
- target/obstacle mutation;
- non-finite state;
- environment/evaluator termination disagreement; and
- an offline history aggregator for two-consecutive mastery and retention.

Before the next promotion run, replace the historical continuous-percentage
retention comparison with the count-based v0.3 contract in K0 and add focused
fixtures for its `n=8`, `n=32`, invalid-evaluation, and streak-reset cases.

Do not turn this into a generic evaluation framework. The output remains one
versioned JSON record per checkpoint and stratum.

Acceptance:

- `mastery_gate.passed` cannot be true with an integrity failure;
- a single-checkpoint result is not labeled two-consecutive mastery; and
- all gate inputs can be recomputed from the saved JSON.

### C5 — Make reward and termination histories auditable

Status: complete.

Verified implementation receipt, 2026-07-25:

- Terra commit `f6bfc007` exposes pre-reset timeout, action-effect,
  productive-workspace-cycle, mass-residual, immutable-target/obstacle, and
  exact manifest-provenance diagnostics;
- terra-baselines commit `475ae47` carries per-environment episode state
  across PPO rollout boundaries and writes one bounded
  `terra_training_episode_aggregate_v1` JSON grouped by stage, family, primary
  cell, and separate `task_done`, `timeout`, `both`, and `other` reasons;
- terra-baselines commit `6c56525` additionally reduces mass residual and
  immutable-target/obstacle mutation over every transition and every device,
  aborting before any checkpoint even when no episode terminates in that PPO
  update;
- additive values use global device sums while integrity maxima use global
  maxima; W&B receives only reduced totals and rates, never an arbitrary
  environment element;
- fixtures cover a two-window episode, population-equivalent shard reduction,
  all terminal labels, and checkpoint-blocking mass, mutation, and reward
  reconstruction failures;
- the full terra-baselines suite passes 130 tests after the F0 launch and
  checkpoint-lineage gates; the final focused aggregate suite passes 5 tests;
- a one-update strict-F0 terminal-path CPU smoke at horizon one records exact
  `foundation / all_around_low_volume` provenance, one timeout, one action,
  return `-0.005` exactly reconstructed by its components, and zero mass,
  mutation, or reward-integrity failures; and
- its exact saved checkpoint reload has 50 finite model leaves, finite
  optimizer state, and `next_update == 1`.

A second real PPO integration smoke used the exact regenerated F0 foundation
manifest with a deliberately reduced CPU shape (one device, two environments,
one step, one update). It exercised the final per-transition hard abort and
checkpoint schema, recorded exact
`foundation / all_around_low_volume` provenance, and produced zero transition
integrity failures. This is implementation evidence only; it does not replace
either production-shaped four-GPU F0 smoke.

Machine-readable smoke receipt:

- aggregate:
  `.artifacts/terra_curriculum_recovery_20260725/c5_terminal_smoke/episode_aggregates/c5-terminal-smoke_update_000001.json`,
  SHA-256
  `29f4cdc2910f43a52781392be19a651883b3243ddd225079ccc5ca5e6cd5ed91`;
- checkpoint:
  `.artifacts/terra_curriculum_recovery_20260725/c5_terminal_smoke/c5-terminal-smoke_FINAL.pkl`,
  SHA-256
  `b57789dbca7fca20ff6e5cb8144444c89d5920f3d4d86f782b774c2b6f46c60a`.
- F0-path CPU checkpoint:
  `.artifacts/terra_curriculum_recovery_20260725/f0_launch_cpu_smoke/f0-launch-cpu-smoke-local-2026-07-26-01-30-27_FINAL.pkl`,
  SHA-256
  `4c67c470cbba240f6ddf00bac7f09aaffb4892fc0e38fc956ca35d664d8c3f0b`;
- F0-path CPU aggregate:
  `.artifacts/terra_curriculum_recovery_20260725/f0_launch_cpu_smoke/episode_aggregates/f0-launch-cpu-smoke-local-2026-07-26-01-30-27_update_000001.json`,
  SHA-256
  `5ca79f8bdd1e37030f4374923a631f75a3fe1ae58e6f51b21b5b41802a1e77fa`.

C5 corrective amendment, 2026-07-26:

- the first foundation production run exposed a false hard failure in schema v1:
  reward and reward components were accumulated independently for as many as
  450 signed float32 steps and only compared at episode end;
- terra-baselines `c58ad23` changes the hard invariant to reward reconstruction
  on every transition, while retaining the independently accumulated
  episode-level difference as an explicitly informational drift metric;
- `terra_training_episode_aggregate_v2` records both quantities separately, and
  the checkpoint-blocking gate uses only
  `step_reward_residual_violation_count`;
- a deterministic 450-step regression with 225 rewards of `+0.8`, 225 of
  `-0.8`, and exact `0.005` existence components reproduces a v1 episode drift
  violation while every per-transition residual is exactly zero;
- a true per-transition missing-component case still fails;
- the full terra-baselines suite passes 131 tests with 69 warnings, the focused
  aggregate suite passes 6 tests, shell syntax, byte-compilation, and critical
  Ruff checks pass;
- a real reduced F0-path PPO checkpoint reloaded with 92 finite model leaves,
  185 finite optimizer leaves, schema v2, and zero transition-level violations:
  checkpoint SHA-256
  `ec48f2058d64f7888745068f4bbff91ac66c17ddf1c9566997ccafea417c1064`
  and aggregate SHA-256
  `4475d5ffc870a64255fd426329e876c8af7f87c4112bf6630e8675d184a6171e`;
  and
- a separate horizon-one PPO smoke forced one completed timeout episode whose
  return and components reconstruct exactly, with zero mass, mutation, or
  step-reward violations: checkpoint SHA-256
  `8805fcf070776b95672d3e025cb0742801348ef6dc0aacd2ea473b45633c71b8`
  and aggregate SHA-256
  `6cc302e0a03a19212050f20962e8206ab7078e6777e047192821322074693411`.

This amendment repairs the receipt gate; it does not change PPO, reward values,
map identity, reset, model, or the preregistered F0 feasibility treatment.

The completed runs expose a logging defect: scalar reward-component fields are
taken from the final state of one environment, while terminal completion fields
cover device 0 and pool successes with timeouts.

Add one bounded full-rollout aggregate keyed by family, primary cell, active
stage, and termination reason. Reduce across every environment and device
before host logging. For each key, preserve raw count and sum fields for:

- exact-target, accepted-mask, buffer-only, and illegal dump volumes;
- dig, dump, and combined completion;
- episodic return and every reward component summed over terminal episodes;
- terminal reward before and after normalization;
- steps, action counts, invalid/no-op actions, and productive workspace cycles;
- mass and immutable-map integrity fields; and
- enough raw counts to recompute every W&B rate offline.

Carry reward-component and episode-stat accumulators per environment across
rollout boundaries. Snapshot and reset them only on that environment's
terminal transition; otherwise an episode spanning two PPO rollouts is
silently truncated.

Do not log an arbitrary environment element as an aggregate. Save the bounded
aggregate as machine-readable JSON at each logging interval and use W&B only
for its reduced totals and rates. Per-map, per-episode receipts remain a fixed
evaluator responsibility; do not stream every training episode or build a
generic event-logging framework.

Acceptance:

- global counts agree with a single-device reference fixture;
- a fixture spanning two rollout windows produces one complete episode sum;
- success, timeout, and simultaneous success/timeout are separately labeled;
- reward-component sums reproduce total episodic return within tolerance;
- no field silently pools success and timeout; and
- all fixed-evaluator and training receipt fields use C1's completion source.

## 7. Phase O — test observation sufficiency before changing the model

### O0 — Construct paired-state alias tests

Do not block the first corrected fixed-identity probes on a broad observation
audit. Trigger this task when a direct paired state is already known or when an
F0 failure implicates missing state rather than transition or reward behavior.

Try to construct current-model inputs that are identical while relevant state
differs:

1. positive pile height `1` versus `>1` outside the local workspace;
2. identical spatial state with different remaining horizon;
3. different last-dig or loaded-soil provenance with different next-step
   legality or reward; and
4. disabled reachability information on an obstacle-constrained map.

For an alias to count, show both:

- equality of every tensor returned by `obs_to_model_input`; and
- a difference in transition legality, reward, or a clearly defined optimal
  action.

Decision:

- if no consequential pair is found, retain the current observation;
- if raw positive height is consequential, change only its bounded
  preprocessing while preserving channel shape;
- add remaining budget or provenance only when its paired test succeeds;
- enable reachability only in a separate treatment; and
- do not add recurrence for state that can be represented directly.

Acceptance:

- deterministic regression tests capture every confirmed alias; and
- any observation-v2 change has a checkpoint compatibility or explicit
  checkpoint-growth decision.

## 8. Phase F — prove dynamic feasibility

### F0 — Overfit one easy foundation and one easy trench

Dependencies: C0-C5, C1a, and C1b. A broad O0 audit is not a prerequisite for
this bounded probe.

Select and visually record:

- one regenerated low-volume foundation with all-around dumping; and
- one regenerated low-volume straight trench with broad both-side dumping.

These are new quantitative starter identities under the corrected contract,
not reused historical M0 maps.

Run two independent policies, one per identity. For each:

- initialize a base `resnet_spatial_8x8` policy from scratch;
- use an independent declared initialization seed, fresh optimizer, and fresh
  schedule;
- use only `corrected_dense_v1`;
- use full 450-step resets with `env_steps == 0`;
- use no partial resets, map curriculum, reward curriculum, or architecture
  change;
- save a checkpoint every 100 updates, then evaluate all ten checkpoints
  deterministically on the same 32 declared reset seeds; and
- run the preregistered 1,000 PPO updates. This first implementation does not
  claim online early stopping. Extend once to 2,000 only if no two-checkpoint
  pass exists and the fixed-seed curve is still improving.

At the current four-device, 1,024-environment-per-device, 32-rollout-step
shape, one update is 131,072 global transitions; 1,000 and 2,000 updates are
131,072,000 and 262,144,000 transitions. Record both units and recompute them
if the probe shape changes.

Frozen F0 launch receipt:

- implementation: terra-baselines `6c56525`;
- foundation initialization seed: `2026072601`;
- trench initialization seed: `2026072602`;
- common evaluation reset seeds: integers `2026072600` through `2026072631`;
- one independent four-RTX-4090 job per identity, with 1,024 environments per
  device, 32 rollout steps, two update epochs, and 32 minibatches;
- explicit learning rate `3e-4`;
- entropy coefficient cosine-annealed from `0.15` to `0.005` over 950 updates,
  so the bounded probe reaches its low-exploration regime before update 1,000;
- base `resnet_spatial_8x8`, float32 encoder, flat minibatch shuffle, no value
  clipping, and finite checks every update;
- no resume, warm start, teacher, map-stage transition, reward-stage
  transition, or partial reset;
- one exact production-shaped update-1 smoke per arm before its production
  command, including reload of the saved model and optimizer plus validation of
  the C5 aggregate;
- 1,000 mandatory per-update aggregate receipts, ten periodic checkpoints, and
  one final checkpoint before a training-complete marker can exist; and
- the evaluator rejects a checkpoint whose seed, treatment, optimizer lineage,
  model, integrity receipt, cadence, or map identity differs from this frozen
  declaration.

The new remote root is
`/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0`.
It must contain immutable source and bank SHA-256 manifests before submission.
The two training arms may run concurrently because they answer independent
fixed-identity feasibility questions. Their dependent evaluators may run only
after the corresponding training job completes successfully.

Submission receipt, 2026-07-26 01:47 CEST:

- launch receipt SHA-256:
  `bc1ab0f3808727b2e0ce13305860a95d7dbe1b4f167f2688642b75dc47fab6e4`;
- submitted-jobs receipt SHA-256:
  `975c0852391b96a73a2f1aa202856b52214811406217351d50adf27fa3832bb4`;
- foundation train job `8629884`, requesting exactly four RTX 4090 GPUs,
  four CPUs, and 32 GB;
- trench train job `8629885`, with the same independent resource request;
- foundation evaluator `8629886`, `afterok:8629884`, requesting one RTX 4090;
- trench evaluator `8629887`, `afterok:8629885`, requesting one RTX 4090; and
- both training jobs were `PENDING (Priority)` at the first scheduler audit.

Submission is not a passed smoke or a training result. Keep F0 and checklist
items 7-8 open until the corresponding machine-readable gates exist.

Foundation update-1 GPU smoke receipt, job `8629884`:

- runtime preflight saw exactly four RTX 4090 devices and passed cuDNN and NCCL;
- the exact 4 x 1,024 x 32 update ran from seed `2026072601`;
- the FINAL checkpoint reloaded with 92 finite model leaves, 185 finite
  optimizer leaves, `next_update == 1`, and the complete frozen config;
- transition mass residual, target mutation, and obstacle mutation are all
  zero;
- smoke gate SHA-256:
  `daa0257ba1e17a195ea8654d9ebbdfbe9344d9b385d8a258990cb3ba0aeb2c36`;
- FINAL checkpoint SHA-256:
  `68b5ab8f6de6a5c211a4dbe407a831aa7e993e888237f99ab3d810a708a2d2ba`;
- update-1 aggregate SHA-256:
  `a5e8b4ef26fa97f1a2f0f870cfe7fef9b68d9c4c72459d33dbfa7733b9473b33`;
  and
- production continued as W&B run `u7hhtnrh`.

The independent trench update-1 smoke in job `8629885` also passed the same
four-RTX-4090 runtime, finite-checkpoint, configuration, and transition gates:

- smoke gate SHA-256:
  `5ba0469aa8d5ef2b7faeffe86461682c194e9d7c60d241575c23f0397a2ea888`;
- FINAL checkpoint SHA-256:
  `69c95ba12de91e7260520c7800d6d2be88b2c7955b4ba9ba01423100417adedc`;
  and
- update-1 aggregate SHA-256:
  `f0b8a2fa7aa3055abb49c0748deb3ec8ce52b4ac07613a5f96b012446455ae7b`.

First-attempt incident receipt:

- foundation production job `8629884` completed 56 valid updates at roughly
  30.5k global transitions/s, then failed before update 57 could be written or
  before the first update-100 checkpoint;
- the sole reported failure was
  `reward_residual_violation_count=1`; log SHA-256
  `5bc005ec3172f3b9fc51713e5b06f8246b18a4dacd99457b04f7195ef5da2d1e`;
- the deterministic long-episode regression above proves this was a C5
  float32-association false positive, not evidence of a missing reward component
  or failed map feasibility;
- dependent foundation evaluation `8629886` was cancelled by `afterok`;
- trench job `8629885` was intentionally cancelled after its smoke and before
  production execution under the known-defective v1 gate; dependent evaluator
  `8629887` was cancelled; and
- all first-attempt source, logs, smokes, and 56 foundation aggregate receipts
  remain preserved under the original immutable F0 root.

No first-attempt production result is admissible as F0 evidence. Replacement
jobs must start from scratch at the same declared seeds and treatment under
terra-baselines `c58ad23`, use a distinct immutable `f0_retry1` root, and repeat
both exact production-shaped update-1 GPU smokes under schema v2. Checklist
items 7-8 therefore remained open until the retry evidence below.

Corrected retry submission receipt, 2026-07-26 02:33 CEST:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0_retry1`;
- Terra revision `188362cd34de6757bc93fff1931561e173a9f2a8`;
- terra-baselines revision `c58ad233e4478316e8aded57361bc29850af6317`;
- source-manifest SHA-256:
  `d1d28bedecfe26767fe902664a76d11bad82bd53c5968b9308416a5152856787`;
- unchanged bank-manifest SHA-256:
  `0e02471987700460b37e42a88bd13548a5cc68ce812f3986514a2dffd3c53d2b`;
- bank-validation SHA-256:
  `40cab18be4527490e531c6289236077e553ce4e322ddb4ebe379d8b56c5c51cc`;
- launch-receipt SHA-256:
  `43681f81faccb8729938b8df07eb90b29f7ffa3421f085bcde16ee40ad751a60`;
- submitted-jobs receipt SHA-256:
  `543a88804305a0350b30d0659cef75922055b2239938d18e5c85663e23a2fc3c`;
- foundation train/eval jobs `8632268` and `8632273`;
- trench train/eval jobs `8632271` and `8632307`; and
- both training jobs were `PENDING (Priority)` at the first scheduler audit,
  with evaluators held by `afterok`.

The launch scripts pass the retry root explicitly through Slurm and override the
log path at submission, preventing an accidental write back into the
first-attempt root.

Corrected retry update-1 GPU smoke receipts:

- foundation job `8632268` passed the exact four-RTX-4090,
  4 x 1,024-environment x 32-step update under aggregate schema
  `terra_training_episode_aggregate_v2`;
- foundation smoke-gate, reloaded FINAL checkpoint, and update-1 aggregate
  SHA-256 values are
  `d193f9e9d94b9299664dab235a2cdcf074b4b0ad39d1cde3c4c9ab6005c60adb`,
  `072f9bd402dc8be697c578058a4b85f32fd72e1bf21fc9e2bc27dd219d00bb82`,
  and
  `921cf537c1a62e4634fdc12adb46d22a5e16ae7430c81004e49699c20a347308`;
- trench job `8632271` independently passed the same production-shaped gate
  with seed `2026072602`;
- trench smoke-gate, reloaded FINAL checkpoint, and update-1 aggregate SHA-256
  values are
  `9e87b0fc93f14bb36d61e4150ee175fcf4b6906d429638dc5b84e63647d39b06`,
  `713d1f9761faa6f33e3b95c31ee7db8f99c4dc6051a831f93ba1d5536059c60a`,
  and
  `7102bfc8f1dbce1248d2265082e996de16f32fb6e1a3fdf099354fed47edbf42`;
- both gates reloaded 92 finite model leaves and 185 finite optimizer leaves,
  matched the frozen configuration and lineage, and reported zero mass
  residual, target mutation, and obstacle mutation; and
- both jobs then entered fresh 1,000-update production runs. No production
  checkpoint or feasibility result is inferred from the smoke.

Corrected retry terminal training receipts:

- foundation job `8632268` completed 1,000 updates in `01:34:41` with exit code
  `0:0`; trench job `8632271` completed in `01:31:48`, also `0:0`;
- each arm has exactly 1,000 schema-v2 aggregates, checkpoints at updates
  100–1,000 in increments of 100, one FINAL, and `TRAINING_COMPLETE`;
- independent terminal gates reload both FINAL and update-1,000 with 92 finite
  model leaves and 185 finite optimizer leaves, prove their model and optimizer
  trees equal, and verify every hard aggregate field;
- foundation and trench training-gate JSON SHA-256 values are
  `800ea2648faaa584988523647576b9d7d93b7c367782bc59ef4facc26380ae19`
  and
  `0379637fd80925911c9aee11e75f514c9a4dc718f4128d1b9e64067de00b94c2`;
- foundation FINAL and update-1,000 SHA-256 values are
  `8ab3cf96f8de4543e8cc071e61e68fc60b4adf779acf92358e3d698b54b505c9`
  and
  `7c3344c270a3e0d25d847810a43fe9b996c5778ec480521a20ba9cf051aea00a`;
- trench FINAL and update-1,000 SHA-256 values are
  `ff76ad33b0db7bbc4455affafb66b329ec1ef99529baba1dbc19930e6f2eaa3e`
  and
  `cbdbe766a208002dbf6ed16fef22da810e3ace4957e084020f5ed85e0b512a0c`;
- no transition has a mass, target, obstacle, or per-step reward-reconstruction
  violation. The retained episode-level float drift is informational: 16
  foundation episodes and one trench episode over the entire run;
- foundation records 2,690,368 online successes and 94,750 timeouts; trench
  records only 8 successes and 290,808 timeouts; and
- both arms executed the full declared 131,072,000 global transitions.

Corrected retry fixed-evaluation receipts:

- foundation evaluator `8632273` completed in `00:11:48`, exit `0:0`; trench
  evaluator `8632307` completed in `00:19:07`, exit `0:0`;
- foundation and trench evaluation JSON SHA-256 values are
  `a05286563e8fdd42e51b67b6b3f12304dbbc95b7f2fdfa183ee2b2b45a774ad0`
  and
  `16b5893fbba396dcfa8adc7751384a1207345238e4317172e3b554c12f298479`;
- every one of the 640 fixed rollouts passes reset, mass, immutable-map, finite
  state, completion, and checkpoint-lineage integrity;
- foundation successes at updates 100–1,000 are
  `0, 0, 16, 32, 32, 32, 32, 32, 32, 0` out of 32. Passing pairs are
  400/500 through 800/900, and a legal successful trajectory is saved;
- update 900 is the foundation feasibility witness, SHA-256
  `68a57f34e0e1cc3f806e8746de27a7e607d3852ec18c8aee1656b1a8fb44c721`.
  Update 1,000 is not promotable: deterministic behavior regresses to 0/32 and
  usually performs no excavation despite its finite, lineage-valid state;
- trench is 0/32 at every checkpoint. Its best individual fixed resets reach
  48/66 legal moved units (72.7% completion), while many initial poses make no
  progress or settle into no-effect behavior; and
- no 2,000-update extension is authorized. Foundation already passes, whereas
  the trench fixed curve is flat, its productive-cycle rate declines, and the
  extension criterion requires continued fixed-bank improvement.

F0 adjudication:

- the corrected action/observation/dynamics contract can learn the foundation
  identity, so broad architecture insufficiency and universal PPO failure are
  rejected;
- the foundation terminal collapse is a retention/action-selection failure,
  not a feasibility failure, and must be protected by checkpoint-bounded
  promotion rather than final-checkpoint selection;
- the tested trench treatment fails cleanly. Its partial legal progress rules
  out a completely broken dump transition, but does not yet distinguish
  geometry/pose reachability, local reward incentive, or policy cycling; and
- B0, F1, and every broad descendant remain blocked until the bounded trench
  trajectory/reward diagnosis selects and validates one minimal repair.

Bounded trajectory-diagnosis launch receipt:

- observer-only code is sealed at terra-baselines `9935c67` under
  `f0_retry1/diagnostics/trajectory_v1`; it does not modify either immutable F0
  arm;
- launch-receipt and source-manifest SHA-256 values are
  `75eb0edb621db6d9d7ab4f086b5bd9d08adf561c998f3b976e508428e2bca645`
  and
  `af5338d6b209e24d6a1de70b2f63a87183b4c4f8ca06119775eac776c9c3cb52`;
- the four preregistered replays are foundation updates 900/1,000 and trench
  updates 900/1,000. Each uses the original 32-reset batch and transition RNG,
  for at most 57,600 transitions and zero gradient updates;
- compact traces are retained for the foundation retention control
  `(900, 2026072600)` versus `(1000, 2026072600)`, the trench progressing/stalled
  pair `(900, 2026072600/2026072601)`, and two distinct update-1,000 failures
  `(2026072602/2026072611)`;
- the diagnostic is inadmissible unless all 128 replayed rows match the sealed
  evaluator in success, termination, length, no-effect count, return, and every
  completion field. Only then may policy logits, action-effect opportunities,
  reward components, repeated states, and counterfactual DO rewards select a
  repair; and
- one-RTX-4090 job `8642618` was submitted at `2026-07-26T02:49:32Z`;
  submitted-job receipt SHA-256 is
  `e2062250a9153eda0968250238abd401f476e94dde77c5edfac82dcc893acfe8`.
  It failed after `00:00:26`, exit `1:0`, because the shared Euler venv has no
  `pytest`; runtime and both immutable manifests passed, and no replay began;
- the failed root remains sealed. Retry root `trajectory_v1_retry1` changes no
  treatment or diagnostic logic and invokes the same two focused test
  functions through Python `runpy`;
- retry launch-receipt and source-manifest SHA-256 values are
  `259e78fc95227daca3629126b3d2a5c6f592f437aa7a56882ee3ce7b526f1236`
  and
  `63c103e5e261671a09e6bf8d2d2ea529f903b17100e0b6b11ba52ab57ce948a1`;
  and
- retry job `8642753` was submitted at `2026-07-26T02:52:14Z`; its
  submitted-job receipt SHA-256 is
  `e46e03b26e126c17ae250ed2344f7d442160f8c1c1b75a7ee92a7916318fab1a`.
  It completed in `00:16:40`, exit `0:0`;
- all four replays and all 128 sealed evaluator rows match exactly, including
  zero maximum float error. The 57,600-transition observer gate passed with
  zero gradient updates; output and log SHA-256 values are
  `8b313453fb3c420d2730195252d82c8ad1afb1c9b88d35a888e2fc7ab6115d3f`
  and
  `be49a22907108ec32039e3ed80dcf170e22231ee4b0a3a8b1e2a7aca9e4dc7ed`;
- the successful foundation update-900 control finishes in 21 effective steps.
  Update 1,000 instead alternates 220 forward and 222 backward actions, visits
  only 11 physical states, and never digs. DO would have an effect on 227
  steps, has positive immediate reward advantage on all 227, yet is ranked
  eighth by the policy every time;
- trench update 900 seed `2026072600` makes 20 effective actions, moves 48/66
  units legally, then selects 430 explicit no-ops. Seed `2026072601` makes nine
  effective moves and then selects 441 no-ops; movement and rotation remain
  effective, but it never reaches a DO-effective workspace;
- trench update 1,000 seed `2026072602` cycles through only six physical states
  for 450 effective movement actions without reaching a DO-effective workspace.
  Seed `2026072611` digs 41/66, places 19, remains loaded with 22, then
  oscillates between 218/219 cabin rotations. DO is effect-capable on 441
  steps but selected only three times; 437/438 missed DO actions have higher
  immediate reward, by `+0.270` on average; and
- the 100-update training-receipt bins independently show the trench no-op
  fraction rising from `13.3%` to `92.2%`, no-effect actions from `35.8%` to
  `92.7%`, and productive cycles falling from `6.73` to `2.09` per episode
  while mean return improves from `-7.91` to `-1.11`. The foundation control
  learns while its no-op/no-effect rates fall.

Diagnosis:

- the map is statically valid and supports repeated legal dig/dump progress;
  transition, capacity, mass, and exact-mask contracts are not the blocker;
- no trace requires different hidden outcomes from an identical model input.
  This is not a broad alias search, so it supplies no authorization for O0
  feature or recurrence work;
- the failed policies exploit idling or short motion/cabin cycles after easy
  progress. The foundation control rejects a universal PPO or base-architecture
  failure, but its update-1,000 collapse makes checkpoint-bounded retention
  mandatory; and
- the first implicated treatment is the trench-only absolute distance/alignment
  reward. It becomes less negative as the population settles and return
  improves while task work collapses. This is a causal hypothesis, not yet a
  selected reward: it requires the one-factor F0R ablation below.

### F0R — Remove absolute trench shaping as a one-factor repair

Dependency: the completed failed-trench diagnosis above. Foundation is not
rerun because it already supplies the matched PPO/model feasibility control.

Freeze every F0 trench choice, including:

- the same exact trench identity, reset bank, horizon, initialization seed
  `2026072602`, fresh optimizer, 4 x 1,024 x 32 PPO shape, learning rate,
  entropy schedule, model, checkpoint cadence, and 1,000-update budget;
- `corrected_dense_v1` action rewards, exact completion, terminal reward,
  transition, and integrity gates; and
- independent update-1 GPU smoke, ten fixed evaluations, and the same
  29/32-at-two-consecutive-checkpoints pass gate.

Change exactly one environment field: set `apply_trench_rewards=false`. This
removes the absolute per-step distance/alignment term; it does not add progress
reward, change action costs, change the map, alter PPO, or begin W0a/W1.
Record the treatment as `corrected_dense_v1_trench_absolute_off`.

Implementation and launch seal:

- terra-baselines `b203d8a922742ec62dfd62cd9b2e24cd7b6eaa3e`
  adds the treatment-specific preset, independent smoke/evaluation contract,
  and `terra_f0_training_gate_v1`. The latter reloads every numbered
  checkpoint and `FINAL`, requires exact update-1,000 model and optimizer
  equality, and audits all 1,000 population receipts before marking training
  complete;
- a field-for-field regression test proves that the control and F0R presets
  differ only in their name, description, and
  `maps[0].apply_trench_rewards`; the semantic treatment changes only that
  Boolean. The seven F0 evaluator tests and two terminal-verifier tests passed
  in the sealed Euler source environment;
- immutable run root
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/f0r_trench_absolute_off_v1`
  reuses the exact F0 retry bank without regeneration. Its source manifest is
  `6b32132f67fd33021094119bd6f9ab320290276f5fd4c0d19eddfc3aada71116`,
  its unchanged bank manifest is
  `0e02471987700460b37e42a88bd13548a5cc68ce812f3986514a2dffd3c53d2b`,
  and its preregistered launch receipt is
  `48ebe22b248accfa9915c53db99d1f306f95dcb464681c7131cba4f72ae78bf1`;
  and
- sealed source revisions are Terra
  `200e30d55e7999bb6c2466343c5000bca37be3e1` (tree
  `6ba53bd7e0603ebb2ff5458acd1345e97f14cd62`) and terra-baselines
  `b203d8a922742ec62dfd62cd9b2e24cd7b6eaa3e` (tree
  `b0a32bc07a490c009c4a43c6d33eb39a9d78e958`). Source and bank are
  read-only after their manifests passed.

F0R was submitted at `2026-07-26T05:24:28+02:00`: training job `8643810`
owns the production-shaped update-1 smoke, the 1,000-update treatment, and the
terminal receipt/checkpoint gate; fixed-seed evaluation job `8643812` has an
`afterok` dependency on it. The submission receipt SHA-256 is
`cb5ac9cad89246723fd963edbdeb1e02e8dcddb2a9a819ec2a16eace171ebb07`.
Both jobs were initially pending for priority/dependency, so this records a
valid submission rather than a passed smoke or experiment result.

Job `8643810` subsequently failed in `17 s` on `eu-g6-064` before the source
tests or PPO: Slurm and `nvidia-smi` allocated four RTX 4090s, but the
independent runtime gate saw only JAX devices `0`, `1`, and `3`. Evaluator
`8643812` was therefore cancelled by dependency. The untouched failure log is
`c0b9e513c2cd5e4da03ca4ac8e66633d27457b2276f5ba8f485accc80137de24`;
this is infrastructure evidence, not an update-1 smoke or treatment result.

The no-treatment-change retry uses immutable root
`f0r_trench_absolute_off_v1_retry1`, launch receipt
`49a8e0ae582fbfb45075cc841ca4fd4ffbe2ebe1af9e295ee8513c7a6c557e46`,
and the same source/bank manifests. Training/evaluation jobs
`8643823`/`8643824` were submitted at `2026-07-26T05:26:31+02:00`;
their submission receipt is
`8c7431a0a5c44723ac7ea48637692870406576e99a5369cf40a9d542ba9a6668`.
Both pending jobs have `eu-g6-064` in the scheduler's explicit exclusion list.

Retry job `8643823` then passed the four-device CUDA/cuDNN/NCCL preflight, all
nine source-contract tests, and the exact saved update-1 smoke on
`eu-g6-062`. `SMOKE_GATE.json` is
`172a86f1449d5f10241480cd279a68fc2f1d42352e1641348fd167bafbd35dd5`;
it reloads checkpoint
`2a3779a326a045d346e3918e2dbde297b1ac90d93a0a6a83a65402954a106a2f`,
checks 92 finite model and 185 finite optimizer leaves, verifies the exact
treatment/configuration, and reports zero mass, target, or obstacle
violations. Its update-1 population receipt is
`307556881f098f99972e52538876cc290b674951494e105ad8b370166dc310f7`.
The independent 1,000-update production initialization then started as W&B run
`nosra33p`; neither smoke completion nor a running production job is an F0R
mastery result.

At update 100, the first numbered production checkpoint
`0efaf2e7dd2e5ea185098314a6ecc18b9e02eb8dc4a93e1802b1b704027b3cfc`
independently reloaded on CPU with the exact treatment/configuration, 92 finite
model leaves, 185 finite optimizer leaves, and zero checkpoint transition
integrity counters. Updates 1-100 contain 28,674 completed online episodes,
81 successes, and no hard aggregate failures. The synchronized timeout
cohorts' mean completion rose from `0.293` at update 15 to `0.584` at update
99, while no-op/no-effect rates remained about `13-15%`/`34-35%` instead of
forming the historical late idling attractor. These are healthy intermediate
training signals only; fixed greedy mastery remains unmeasured.

The online treatment response is material by checkpoint 400. Checkpoint 300 is
`4dc6b365851a300244a9b4fbb48457ab091b76c992bdda9050ca0ebb03fbe0fb`;
updates 201-300 completed 130,602/138,269 online episodes successfully
(`94.5%`) while no-op/no-effect rates fell to `11.9%`/`26.5%`. Checkpoint 400
is `fbbb23faf80e217b3ef0f78772532ef109580720d66191f82161f93306c722f2`;
updates 301-400 reached 334,558/334,793 online successes (`99.93%`) with
`4.9%` no-ops, `10.6%` no-effect actions, and zero hard failures. Thus F0R
removes the measured online idling attractor, but only the complete
ten-checkpoint fixed evaluation can establish mastery and late retention.

F0R final adjudication: **PASS**.

- training job `8643823` completed 1,000 updates in `01:34:25`, exit `0:0`;
  fixed evaluator `8643824` completed in `00:11:19`, also `0:0`;
- terminal training receipt
  `212f2da2f8cd7c1305f9ad7c500f7ba5e4964e17bdd5cd9685bd3296e0f868fb`
  certifies ten checkpoints, all 1,000 schema-v2 aggregates, 92 finite model
  and 185 finite optimizer leaves, and exact `FINAL`/update-1,000 model and
  optimizer equality. The checkpoint and aggregate manifest hashes are
  `ba628a2542758c34933f4c8e3deba3e2599d50ad6cd93ba94127eea63196b899`
  and
  `e677589fd5bb810ea7db55dae862bcc15848731efb34f3fe9fadbe00472e30dd`;
- `FINAL` and update-1,000 SHA-256 values are
  `e236db0ee10583a1ad1b50cf2e303791c0668bfc27f98eaecc7dc87b066254f0`
  and
  `d2b0e0bb2a36c686692ce2ba6bbdc0e5db445563ea9b69952f104b50cff4be53`.
  Across 2,892,988 online episodes, the treatment records 2,820,943 task
  completions and 72,054 timeouts. All mass, target, obstacle, and per-step
  reward-reconstruction gates are zero; four episode-sum drift flags remain
  informational under the frozen C5 contract;
- final fixed-evaluation JSON
  `1a4cc49fa8d6d5a1cb53c8c0ad98e63bf144a40b38f6de9e0f1f0865987ba672`
  evaluates the exact ten-checkpoint sequence on the 32 frozen resets. The
  success curve is `0, 7, 31, 32, 0, 32, 32, 32, 32, 32`, with zero integrity
  failures at every checkpoint;
- the required consecutive gate first passes at updates 300/400 and also
  passes at 600/700, 700/800, 800/900, and the terminal 900/1,000 pair. The
  update-300 witness solves reset `2026072600` in 35 legal effective actions;
  the terminal update-1,000 witness solves it in 30; and
- update 500 is a real isolated greedy-selection collapse (`0/32`, mean
  314.25 no-effect actions) despite strong online behavior. Recovery to
  `32/32` at update 600 and perfect retention through update 1,000 means the
  preregistered gate passes, while independently reinforcing the requirement
  to select qualified checkpoints rather than assume monotonic PPO behavior.

The selected easy-trench parent therefore uses
`corrected_dense_v1_trench_absolute_off`. No extension, O0 observation change,
second reward repair, or W0a/W1 experiment is authorized at this gate. The
retained foundation witness remains update 900
`68a57f34e0e1cc3f806e8746de27a7e607d3852ec18c8aee1656b1a8fb44c721`;
together with terminal trench witness
`d2b0e0bb2a36c686692ce2ba6bbdc0e5db445563ea9b69952f104b50cff4be53`,
it releases B0 to build and validate the orthogonal feasibility panels.

Decision:

- pass: use the shaping-off contract for the easy trench cells and proceed to
  B0; do not call the historical absolute trench term part of the corrected
  parent;
- fail with an improving fixed curve: use F0's single conditional extension;
  and
- fail flat: stop again and test a bounded completion-delta/potential reward,
  not architecture, map diversity, or a broad hyperparameter sweep.

Pass gate:

- at least 29/32 successes in two consecutive evaluations;
- zero integrity failures; and
- at least one saved legal action trajectory.

If a probe fails, stop broader training and classify the failure:

- no legal action sequence within the horizon;
- action mask or transition blocks required work;
- reward gives the wrong local incentive;
- observation alias;
- PPO/value instability; or
- map itself violates the intended dynamic-feasibility contract.

Single-map success proves only that the current dynamics can learn that
identity.

### F1 — Train easy family-bank specialists

Dependencies: the retained foundation F0 witness and the shaping-off trench
F0R witness pass, C5 passes, and B0c has expanded the eight dynamically
witnessed primary easy cells into immutable large banks.
Family generalization must not be judged from the current historical
eight-identity-per-cell M0 pool or from the small B0a feasibility panels.

Train two independent base `resnet_spatial_8x8` policies from scratch:

- `EASY-FOUNDATION-SPECIALIST`;
- `EASY-TRENCH-SPECIALIST`.

Hold PPO, model, reward, horizon, reset, and evaluation fixed. Use only the
named family as the treatment. Evaluate deterministically on the
source-disjoint quantitative easy-family bank every 100 updates.

Use the global milestone rule: review at 1,000, 2,000, and 5,000 updates,
advance whenever even slight fixed-bank task progress remains, and stop the
screen once the family gate passes twice. Each family immediately receives its
own at-least-20,000-update continuous production run on `gpuhe.120h` after
qualifying; it does not wait for the other family. Those long runs may execute
concurrently with the remaining specialist screen and, once both specialists
pass, with G0. They are not substitutes for G0's multitask gate.

Family pass gate:

- at least 26/32 family successes;
- at least 6/8 in every primary cell;
- two consecutive evaluations; and
- zero integrity failures.

Interpretation:

- both pass: the families are learnable separately; multitask interference or
  mixture design becomes plausible;
- one fails: diagnose that family before a generalist;
- both fail: do not run the map curriculum.

These specialists are feasibility instruments. Call each a teacher candidate
only after it passes; neither is yet the final multitask teacher or deployment
policy.

## 9. Phase R — explain the historical M0 regression

### R0 — Fork the flat update-2,000 checkpoint

Run only if D1 does not show that the historical semantic mismatch dominates
the measured trajectories and both F0 probes pass.

Use the exact historical Terra and terra-baselines revisions and the full
update-2,000 checkpoint, including optimizer state. The historical control
falls from 22/64 at update 2,000 to 15/64 at update 2,500, making a 500-update
fork the shortest useful reproduction window.

Arms:

- control: continue the historical terminal mixture;
- treatment: continue M0 only.

Both arms restart environment/RNG state through the identical resume path and
use paired seeds. Do not include the corrected dump contract, observation
change, entropy change, or partial resets.

Evaluate every 100 updates:

- unique train identities and development M0;
- foundation, trench, and every primary cell;
- M1 at fork start and finish;
- policy KL and greedy-action disagreement from the parent;
- action-logit margin;
- entropy, actor loss, value loss, and explained variance.

Preregistered decision:

- support heterogeneous-exposure regression if the control reaches `<=16/64`
  while M0-only remains `>=20/64` in two consecutive evaluations;
- reject exposure as a sufficient explanation if both reach `<=16/64` or the
  final gap is `<4/64`;
- call it inconclusive if the control remains `>=20/64` and fails to reproduce
  the historical decline.

Budget: 500 updates per arm.

## 10. Phase G — establish a corrected easy multitask parent

### G0 — Train foundations and trenches together on the corrected easy bank

Dependency: both F1 specialists pass.

Initialize a third base `resnet_spatial_8x8` policy from scratch, not from E8
or either specialist. Train a 50/50 easy foundation/trench mixture with the
corrected contract and the expanded B0 training bank.

Evaluate every 100 updates. Review at 1,000, 2,000, and 5,000 updates and
advance under the global slight-improvement rule. Once G0 passes twice, its
recipe starts an at-least-20,000-update continuous `gpuhe.120h` production
run under rule 15, with the treatment held fixed.

Pass gate:

- at least 26/32 foundations;
- at least 26/32 trenches;
- at least 6/8 in every admitted easy primary cell;
- two consecutive evaluations;
- zero integrity failures.

The train-development gap remains a required diagnostic but is not an
additional post-hoc mastery threshold.

If specialists pass and G0 fails, the next treatment is sampling/gradient
interference, not a larger encoder by default.

If it passes, this checkpoint becomes the new-distribution small multitask
teacher. No medium growth or reward curriculum begins before it passes.

### S0 — Grow and qualify one medium student

Dependency: G0 passes.

The existing growth utility is an approximate grown warm start, not yet a
proven function-preserving transform for base `resnet_spatial_8x8` to medium
`resnet_spatial_8x8_se`: widened slices and new SE parameters are freshly
initialized. Before S0, either make that exact or ratify the approximate
contract with a frozen-observation policy-logit/value delta gate and an
explicit threshold. Initialize through `--warm_start_from`, strip inherited
optimizer/update state, preserve parent and growth hashes, and state the new
optimizer and training-schedule semantics in the run receipt.

Train on the exact same corrected easy 50/50 bank used by G0. E8 remains a
zero-shot historical reference and supplies no parameters or distillation
targets.

Evaluate every 100 updates and apply the global 1,000/2,000/5,000 milestone
rule plus the same family, cell, two-consecutive, and integrity gates as G0.
Once qualified, the selected S0 recipe receives its own at-least-20,000-update
continuous `gpuhe.120h` production run under rule 15.

The purpose is to establish one medium parent on the new distribution, not to
compare architectures. A scratch-medium control is conditional on failed
growth/qualification evidence; it is not part of the minimal first set.

## 11. Phase B — increase procedural diversity

> **Historical evidence block.** B0a/B0b results and receipts below remain
> authoritative evidence, but their forward-looking cell names, 64-map B0c
> contract, and "near/far distance ladder" are superseded by K0 and
> `MAP_BENCHMARK_SPEC.md` v0.3. Historical `distance` labels mean measured
> dig/dump separation; they do not prove loaded transport or forced
> rehandling. Nothing in this block authorizes a new launch.

### B0 — Rebuild quantitative cells, then expand source-disjoint banks

Dependency: the retained foundation F0 witness and shaping-off trench F0R
witness pass.

Do not enlarge the failed M0-M2 bank unchanged. Its all-around and large-apron
foundation cells differ by roughly an order of magnitude in relative dump
area, its procedural foundation cells change geometry and dump layout
together, and its topology/site cells are not a monotonic ladder.

#### B0a — Build small paired feasibility panels

First build small paired feasibility panels that change one axis at a time:

- OSM versus procedural foundation geometry under identical all-around dumping;
- foundation and straight-trench broad-apron dump distance centered near 2, 4,
  6, and 8 tiles under fixed geometry, volume, capacity, and site;
- straight-trench close side access as a paired broad-both-side versus
  broad-one-side treatment with identical geometry, volume, capacity, and site;
- straight, two/three end-to-end segment, T, X, and disconnected trench
  topology under easy side-cast dumping; and
- site constraints only after the corresponding geometry/dump cell passes.

The exact B0a candidate-cell names are:

```text
foundation geometry:
  f_osm_all, f_procedural_all

foundation distance:
  f_apron_d02, f_apron_d04, f_apron_d06, f_apron_d08

trench distance:
  t_straight_both_d02, t_straight_both_d04,
  t_straight_both_d06, t_straight_both_d08

trench side access:
  t_straight_both_d02, t_straight_one_d02

trench topology:
  t_straight_both_d02, t_segmented2_both_d02,
  t_segmented3_both_d02, t_T_both_d02, t_X_both_d02,
  t_disconnected_both_d02
```

The repeated anchor cells are one immutable dataset identity set referenced by
more than one panel, not independently regenerated lookalikes.

Use eight unique train and eight source-disjoint development identities per
candidate cell. Within a paired distance or side-access panel, the same source
geometry is deliberately reused across conditions and receives one explicit
`paired_source_group_id`; source geometry and generator seeds never cross the
train/development boundary. Exact target-array duplicates are forbidden.

Remote haul at 12 or more tiles remains a separate conditional feasibility
track.

Every starter and panel map uses the exact visible accepted mask, at least
`3x` reachable single-layer-equivalent capacity, and no obstacles unless site
constraint is the isolated axis. Validate capacity under C1a's contained-pile
and numeric-range rules; a large 2-D area ratio alone is not sufficient if a
valid bucket sequence can overflow the stored height type.

Then use offline procedural generation; do not add online generation or PLR.

Initial target:

- at least 64 unique training identities per primary cell;
- no repeated training slots;
- separate fixed promotion, development, and sealed banks with eight maps per
  primary cell each;
- disjoint generator seeds and source geometry IDs across every split; and
- frozen quantitative geometry, dump-distance, reachable-capacity, and site
  contracts for every admitted cell.

For each cell, save:

- geometry and junction/component metadata;
- dump side/components, capacity ratio, and shortest-path statistics;
- site constraints and work volume;
- generator revision and seed;
- source ID and split; and
- accepted dump-mask definition.

Validation:

- generate a random contact sheet per cell for visual review;
- reject templated duplicates using exact hashes plus a declared geometric
  similarity check;
- run static capacity/access validation; and
- measure the actual requested distance statistic rather than labeling a
  generator parameter as the achieved bin.

Static acceptance of B0a requires:

- all arrays and metadata pass the exact Terra loader;
- all train/development source sets are disjoint;
- the only repeated dig geometry is a declared within-split paired source;
- every map passes C1a contained-pile capacity and numeric-range validation;
- every distance cell's dig-boundary-to-accepted-dump median is within
  `0.75` tile of its declared 2/4/6/8-tile center;
- the one-side trench cell has no accepted cells on the forbidden side, while
  the both-side anchor has material accepted capacity on both sides;
- topology metadata and connected-component counts match the saved raster; and
- all galleries, manifests, generator/source hashes, rejection counts, and
  validation receipts are sealed before PPO.

B0a result: **PASS** at `2026-07-26T08:00+02:00`.

- implementation commit: Terra `eaf9cf00`;
- canonical inspectable bank:
  `/home/lorenzo/moleworks/.artifacts/terra_b0a_paired_panels_20260726`;
- builder SHA-256:
  `3a1bb66798f6a4bfc1dc5b3515c5a4485eb9a28d6ffe7c9e8413a548492b79a9`;
- identity manifest SHA-256:
  `911b6e3a453d6d9e1aeaebfe5fcef33406c89aae0180e1c4eb8739efc1fd5b4e`;
- source registry SHA-256:
  `1ffe22f8c3ed4cc608fc8fc9a5106f2ecd26d8e122d042d0d2630b025a293a8d`;
- validation SHA-256:
  `aeebafae74f77d19a11617f464969f162c82385aa52a65ff246f0227bd731ca5`;
- complete file-manifest SHA-256:
  `89a5b5325e4e6872f7899b087ac5d0a8cd444dac30315feee4f342f8e532a347`;
- 256 unique identities and target arrays cover 16 cells, eight train plus
  eight development identities per cell; all 42 per-cell/panel directories
  reloaded through the exact Terra loader;
- all 32 declared paired source groups preserve the exact dig raster, while
  train and development source sets are disjoint;
- the generator rejected one exact dihedral straight-trench duplicate before
  sealing; accepted within-cell maximum dihedral IoU ranges from `0.439` to
  `0.936`, below the declared `0.995` ceiling;
- constrained maps provide `3.25-3.26x` single-layer-equivalent capacity.
  Achieved p50 path-distance ranges are `2.00`, `3.83-4.04`,
  `5.66-6.24`, and `8.00-8.16` tiles for the declared 2/4/6/8 cells;
- every one-side target has zero forbidden-side cells and every both-side
  target reserves at least 40% of its cells on each declared side;
- the five panel galleries and all 32 cell galleries were visually inspected;
  no obstacle/site axis is present; and
- four focused generator tests plus eight contained-transition/loader tests
  pass (`12 passed`). The file manifest verifies without error.

This is static and loader evidence only. B0 remains unchecked until B0b
supplies legal dynamic trajectories and B0c expands the eight witnessed easy
cells.

#### B0b — Five bounded dynamic panel witnesses

Run five independent scratch base-small specialists, each changing only the
named map panel:

| Run | Training cells |
|---|---|
| `B0-GEO-F` | the two foundation-geometry cells |
| `B0-DIST-F` | the four foundation-distance cells |
| `B0-DIST-T` | the four trench-distance cells |
| `B0-SIDE-T` | the paired close both-side/one-side trench cells |
| `B0-TOPO-T` | the six trench-topology cells |

Use `corrected_dense_v1` for foundations and
`corrected_dense_v1_trench_absolute_off` for trenches. Hold PPO, architecture,
450-step untouched resets, and every non-map setting at F0R. Each run receives
500 updates initially and deterministic development evaluation every 100
updates. For an unpassed cell, advance through the 1,000/2,000/5,000
milestones whenever the global slight-improvement rule passes. Because
checkpoint v2 cannot preserve environment/RNG/action-history state, the first
500-to-1,000 advance is a new continuous 1,000-update scratch confirmation
with the same panel recipe and declared seed, not a resume. The 500-update
runs and their best checkpoints remain immutable evidence. A passing panel
stops because its purpose is only a dynamic witness; long 120-hour-queue
training begins after the family recipe is qualified.

A candidate cell earns a dynamic witness only when:

- at least 6/8 development identities succeed at two consecutive scheduled
  checkpoints;
- all transition, completion, termination, and mass-integrity fields are zero;
  and
- at least one successful legal action trajectory for that cell is saved.

The panel policy is a feasibility instrument, not a curriculum parent. If one
cell fails while another cell in the same panel passes, run at most one
conditional scratch single-cell specialist for the failed cell before calling
it dynamically unproven. This conditional run uses the same budget and gate.
Do not launch all cells as an unconditional hyperparameter sweep.

Frozen B0b implementation receipt:

- terra-baselines revision `c42aa61` defines the five panel presets, scratch
  seeds `2026072701` through `2026072705`, exact update-1 smoke, 500-update
  training gate, deterministic development evaluator, and paired Slurm
  launcher;
- all five runs use 4 x RTX 4090, 1,024 environments per device, 32 rollout
  steps, base `resnet_spatial_8x8`, float32 encoder compute, and the full F0R
  PPO/reward treatment except for the declared panel and independent seed;
- each panel retains checkpoints every 100 updates and evaluates every
  development identity with recorded legal action traces;
- the evaluator emits exactly one of `panel_witness_passed`,
  `continue_same_panel`, `conditional_cell_isolates`, or
  `stop_and_diagnose_panel`;
- `continue_same_panel` requires a preregistered task-metric improvement within
  the last five scheduled evaluations. A reward, loss, or online-success
  change cannot authorize more compute; and
- six B0 evaluator/config tests, two training-receipt tests, seven F0
  regression tests, Python compilation, Black, `bash -n`, ShellCheck, and
  whitespace checks pass locally. The production update-1 GPU smokes remain
  cluster gates and are not claimed by these local checks.

Submitted B0b execution receipt at `2026-07-26T08:17:28+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_v1`;
- source revisions: terra `5f7351a8fb13a912c15887265d70359ffe99e976`
  and terra-baselines `7474c3e954386881db435c54495005184649ceb8`;
- source-manifest SHA-256
  `a060c65f0fa33bf791ae4b11974b3276dcec8b51c3c3fd0703fb4cdd895e7f97`
  and bank-manifest SHA-256
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`;
- the remote source manifest, remote bank manifest, and the bank-internal
  manifest all passed before submission;
- submission-receipt SHA-256
  `823c4f21800dc03aa12cd5328f09a35b97cf9e29a215f3bd300ba282ddfcd544`;

| Panel | Training job | Dependent evaluation |
|---|---:|---:|
| foundation geometry | `8647662` | `8647663` |
| foundation distance | `8647664` | `8647665` |
| trench distance | `8647666` | `8647667` |
| trench side | `8647668` | `8647669` |
| trench topology | `8647670` | `8647671` |

This receipt records submission, not a passing smoke or learning result. Each
evaluation remains `afterok`-dependent on its matching training gate.

Infrastructure replacement receipt at `2026-07-26T08:19+02:00`:

- original training jobs `8647662`, `8647666`, and `8647670` all landed on
  `eu-g6-064`, where Slurm allocated four RTX 4090s but the frozen runtime gate
  independently saw only JAX devices `[0, 1, 3]`;
- all three failed before the update-1 smoke began, wrote zero training files,
  and their dependent evaluators `8647663`, `8647667`, and `8647671` were
  cancelled as `DependencyNeverSatisfied`;
- no GPU check was weakened and none of these jobs is scientific evidence;
- identical clean replacements exclude only `eu-g6-064`:

| Panel | Replacement training | Replacement evaluation | Receipt SHA-256 |
|---|---:|---:|---|
| foundation geometry | `8647704` | `8647705` | `0edc025cfcb4f3251cc2a13925f1fb37c32cd8078c56a2d4fcc812c380194e24` |
| trench distance | `8647719` | `8647720` | `fceb4cdaa0e58be811b98ba675dcd9d87a009b4f4165a18bc2bddbaea0b793ca` |
| trench topology | `8647721` | `8647722` | `b9b7fd320812302c0fe4b5543412cc6c5760a1b91cc5f5d33103608d93a64c9d` |

Receipt-gate correction at `2026-07-26T08:28+02:00`:

- the saved scratch checkpoints correctly serialize the inert
  `load_env_from_checkpoint` field as `false`; the first B0 verifier expected
  `true`, even though `resume_from` was `null`;
- foundation-distance job `8647664` and trench-side job `8647668` each
  completed exactly one smoke update, saved model/optimizer/aggregate state,
  then failed only this receipt comparison before production;
- the three still-compiling replacements `8647704`, `8647719`, and `8647721`
  and all remaining dependent evaluators were cancelled once the deterministic
  shared failure was known, avoiding invalid or wasted production work;
- terra-baselines `c42aa61` changes only that serialized-field expectation and
  its unit fixture;
- the corrected verifier independently reloaded both saved family examples:

| Panel | Checkpoint SHA-256 | Aggregate SHA-256 | Corrected receipt SHA-256 |
|---|---|---|---|
| foundation distance | `45ae0b6c4471c11dd6b80b3fb11a5f774ec17ae3af89a3f5897bc8ae9c1a49da` | `d4998464eca8a88a703f7b3703f41404c8da0fb60c39e689f4edd0cb19bfe5b8` | `0f44b9309f4e303b29c079eab32d9d74dd2cb1fe9691fbce02d4eb221dd13782` |
| trench side | `9b0e18060643ca93d2dffbb36df69798662a19d2561e6c2b68e663e6fd66594c` | `1139b4de0fb7e332dc478bccddd705b2911876e36d17c152844ea041208d3d39` | `b381a09331fb218c7c9bbbf40afd212d95f6a842e1c353d451c4116b7b86a4db` |

Both corrected receipts have 92 finite model leaves, 185 finite optimizer
leaves, zero transition-integrity violations, exact manifests, and the frozen
foundation/trench reward contracts. These are valid update-1 smoke witnesses,
not B0 dynamic-feasibility results. The submitted source root remains
immutable; production restarts only from a new root containing `c42aa61`.

Corrected B0b submission at `2026-07-26T08:31:24+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_v1_retry1`;
- source revisions: terra `10cf8f03fb59a94209aa1225148d461d30817fa8`
  and terra-baselines `c42aa612af747156e0ff027a90f8c3db825d8e4a`;
- source-manifest SHA-256
  `f4b0bd57ae9c08955b56357631956da925ef10044a74d353caff40c7242efe58`
  and unchanged bank-manifest SHA-256
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`;
- all source, bank, and bank-internal manifests passed remotely before launch;
- submission receipt SHA-256
  `95ab8f3c28086447716a5e80a4d4c92c2e821deff5d958330ba24faebe18c01b`;
- every training job excludes only the diagnosed node `eu-g6-064`;

| Panel | Training job | Dependent evaluation |
|---|---:|---:|
| foundation geometry | `8648071` | `8648073` |
| foundation distance | `8648076` | `8648078` |
| trench distance | `8648081` | `8648084` |
| trench side | `8648088` | `8648090` |
| trench topology | `8648092` | `8648094` |

This is again a submission receipt only. No B0b result is accepted until the
new-root saved checkpoint, aggregate, training gate, and deterministic
development evaluation pass.

Corrected update-1 result: **PASS for all five panels** at
`2026-07-26T08:42+02:00`.

| Panel | Checkpoint SHA-256 | Aggregate SHA-256 | Smoke-receipt SHA-256 |
|---|---|---|---|
| foundation geometry | `8260b717e9239f8074b9f6fd1455506720a3e9d46f95a789f6f7b3bd69247fb5` | `72d8241ddae904f729f27c0126c68d39f687efe9a8a3f1e291ff2f26f30cd451` | `7e6c9a9d7ece50d77adc9f9f72d5ec04fc9986a33ac39ef518d4b9dbe4536b78` |
| foundation distance | `baa60b77639b34abd8e4d84a0457565bbc77dbf2bbbbe2286359ad4a6948f6ad` | `3b39cc7fb9d77e0ad29ec1c4f563b4c3f8113cbd9d83c78e25af784158846072` | `c256533dc549cfc44c65703111687130806b0cedc76eb05ac13bfeac68b3fbda` |
| trench distance | `f7a6a4ae3463c41a27a05b115ce0c0b67ea477e62ec3475f8adb210baec89c5d` | `37cdca6b460c307bf7aee05c7ed9e023463628f02dd462b55612ff012c8032c3` | `f9941ec6641cb79f92c93517253627232051e0e4aea19374f7e2756230862746` |
| trench side | `09573db08efe194fe227d7cbfc33bbaf3668e16511c47b005fda22f9ebf384ee` | `ea0fe566961f50edc0efb57efb65d1aa99971bf4103964e2d392080943cd6734` | `6f3d1ac117feb1df7e1899568b0a1581ba745c342b209f16c83f1209faef2d92` |
| trench topology | `86dae52e7af64321b182672e8ab29aa19b62d10545499fde3aee105ce31f3bcd` | `6746964f2f2e0d593f8c0248fbde90ed776e3f439b5b5734b37f7bf1660d9da0` | `9155c6c7f58ead49cd2206d30908cae1969407e0b34b6f5ce364e403eceffe2f` |

Every receipt records 92 finite model leaves, 185 finite optimizer leaves,
zero transition-integrity violations, the exact panel manifest, and its frozen
foundation or shaping-off trench reward contract. This passes the launch gate
only; it does not establish a dynamic cell witness.

Bounded 500-update training result: **PASS for all five panels** at
`2026-07-26T09:27+02:00`.

| Panel | W&B | Training-gate SHA-256 | FINAL SHA-256 | Online done / episodes |
|---|---|---|---|---:|
| foundation geometry | `q24yinzc` | `d02a6603e49e6a5c910483741033da7676d9a9b21fd0fe5982f632cda4b5de80` | `d2b259bacc8ecee46526f6b101b0f36f3ac64b51637f0f99495b79c0d2887e98` | 1,632 / 144,167 |
| foundation distance | `rryzgmnh` | `8db63c0643b8f9c2ab7f38bb2d868e75099513eeb9c37a6a11294cb7263fe878` | `8b488c292178504dec7badc37d287e64f48819a1db5451a8b217b16861965fc2` | 7,734 / 146,305 |
| trench distance | `ghs31rt3` | `a8fb706747a400f381c1cd301c7e50dc756510613420c6ec714a8f548dde938c` | `ed7387409192fbd64d7d121528cf8706529959b5c400c414f3d6f8d4afd992de` | 6,314 / 147,559 |
| trench side | `314071nj` | `2053fd48db240d53759c159281ca8ebefc49c47adcc88b618f6b563e87b764c2` | `215d2e9feb1efcf58d0ff077d73ac026cdaf9893dbc059ddf9358d334cf8740b` | 9,281 / 149,616 |
| trench topology | `jwvr6vvy` | `bcc7bff23b94e52a91fd956597867b988d077d137e94326f0ca7ac695d42c1ac` | `cba2fcbca23e2e97b08bfdcb6cfbbec8a4261207089b3ef83422efe045ce7fcc` | 2,408 / 144,577 |

Every training gate has exactly 500 aggregate receipts, numbered checkpoints
at 100/200/300/400/500, one FINAL checkpoint exactly equal to update 500,
maximum mass residual zero, and zero mass/target/obstacle/reward-residual
violation counts. The maximum sub-threshold floating reward residual was
`4.76837158203125e-07`. Online completion rose during every panel but remains
diagnostic; only the running source-disjoint development evaluations adjudicate
B0b or authorize continuation.

Bounded 500-update development result: **all five panels authorize more
bounded compute; none has a dynamic cell witness yet**.

| Panel | Evaluation job | Evaluation SHA-256 | Successful identities observed | Best scheduled median-completion evidence | Decision |
|---|---:|---|---|---|---|
| foundation geometry | `8648073` | `1c101c44d553dc4fbac48d7576ff27d61616948ae0d485804cc839c595d74783` | none | OSM `0.780 @400`; procedural `0.417 @300` | continue |
| foundation distance | `8648078` | `fd4ac9eef8769322ad8052c026061f2426a1aeaf8bf0f951889fd59130cdca46` | none | d02 `0.463 @300`; d04 `0.464 @400`; d06 `0.363 @400`; d08 `0.274 @500` | continue |
| trench distance | `8648084` | `e6308b802ce58cc62e9fa5cc70282cf97721eca01d1b106d968b8f3e7b59631e` | one d02 identity at update 400, with trajectory | d02 `0.448 @200`; d04 `0.428 @200`; d06 `0.452 @200`; d08 `0.567 @300` | continue |
| trench side | `8648090` | `a29f213839a213cf309342792c80228503c6d78bc5ca0af89ccd0678d4623f08` | one one-side identity at update 500, with trajectory | both-side `0.495 @200`; one-side `0.482 @300` | continue |
| trench topology | `8651299` | `9e3154582834080b617e8d44b9e0469e15d2ecd188f9fdb46fbea76840247a3a` | one segmented-2 and one segmented-3 identity at update 500, both with trajectories | straight `0.784 @400`; T `0.561 @400`; X `0.562 @400`; disconnected `0.644 @300`; segmented-2 `0.754 @400`; segmented-3 `0.624 @500` | continue |

All 25 scheduled fixed-bank evaluations have zero integrity failures and exact
reset-manifest verification. No cell reaches 6/8 at two consecutive
checkpoints, so no row is promotable and B0 remains unchecked. Several
terminal checkpoints regress from an earlier best; preserve all checkpoint
histories and select by fixed-bank evidence rather than assuming FINAL is
best.

Topology evaluation recovery:

- original evaluator job `8648094` failed before rollout because 48 maps were
  passed to an inherited 32-minibatch topology, a shape-only evaluator defect;
- terra-baselines `d7867a7` selects `gcd(48, 32) = 16` evaluation
  minibatches, and `d415af2` allows a pinned evaluator source without
  modifying the immutable training root;
- retry source root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_eval_retry1`;
- retry source-manifest SHA-256
  `9daffd0188a7dd5321996bdd5ea1cd4c23447b4a8453bbec75c88fb058cc3a45`
  and submission-receipt SHA-256
  `9ac2068d8f9690999d505b9fefaa18e38c80c8147ba8e4880f59cca9e243754a`;
- retry job `8651299` completed in `00:12:28` on `eu-g6-005`, using the
  unchanged five training checkpoints and bank. Its update-100 through
  update-500 success totals are `0, 0, 0, 0, 2`.

The next bounded treatment is therefore one fresh, continuous 1,000-update
scratch run for each of the five unchanged panel recipes. This deliberately
duplicates the first 500 updates: it preserves continuous process state across
the 1,000-update milestone and avoids falsely describing checkpoint-v2 resume
as exact. terra-baselines `fc93c29` parameterizes the sealed launcher,
training gate, evaluator, checkpoint count, receipt, panel subset, and
diagnosed-node exclusion for the 1,000-update treatment. `bash -n`,
ShellCheck, and whitespace checks pass. Each new run must still pass its own
remote source/bank manifest checks, update-1 GPU smoke, 1,000 aggregate
receipts, ten checkpoint gate, and deterministic development evaluation
before it contributes evidence.

Continuous 1,000-update submission receipt at
`2026-07-26T09:49:32+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_u1000_v1`;
- source revisions: Terra
  `917fbf35a0ffa468148317a42812c7e97b0f5bc1` and terra-baselines
  `fc93c294861781e19e135c82fb645465b4d8e917`;
- source-manifest SHA-256
  `562e3b883282314cc7a6441cd5d7ce614d839f04a9a74379f84f1c46ca9ba955`;
- unchanged bank-manifest SHA-256
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`;
- creation-receipt SHA-256
  `e5e71d8d87c2a618ac03b1b60cc0b6e47c2410744de287e4011ec2b6c2d069df`;
- submission-receipt SHA-256
  `81a7de0bb051165b9fdb52b67d563807ab06f179faa4da49c2961fe21f990b55`;
- the source manifest, bank manifest, bank-internal manifest, exact revisions,
  update target, and executable launch scripts passed before submission;
- every job excludes only the diagnosed node `eu-g6-064`.

| Panel | Continuous training | Dependent evaluation |
|---|---:|---:|
| foundation geometry | `8651897` | `8651898` |
| foundation distance | `8651899` | `8651900` |
| trench distance | `8651901` | `8651902` |
| trench side | `8651903` | `8651904` |
| trench topology | `8651905` | `8651906` |

This is a compute receipt, not a learning claim. Each dependent evaluator is
`afterok`-gated on its matching update-1 smoke and complete 1,000-update
training receipt.

Continuous-run update-1 result: **PASS for all five panels**.

| Panel | Checkpoint SHA-256 | Aggregate SHA-256 | Smoke-gate SHA-256 |
|---|---|---|---|
| foundation geometry | `22a2552778e259af7102959004da83c64911b0df89a6b0a3f347ec3607452dc7` | `517ea36508f89a3d87ae9f8e7353bd1e82bd774d33bead7ab84f452068db3440` | `b4dbc6aabe687e317ac337f5342c8f9a2a0bf9c4b69a10e9a831c6caa03c7ba3` |
| foundation distance | `6bac8d7df0240066070ed12bd1ccc0150061257be140d89c3f00785c6b7b8790` | `22191963a4c7c232a848113ca2a57846f7dffbdd48809d242683148fb7e31bcb` | `50e986d8b954709c8e806da1aeb5229878bab89fb6595e410b87a89f0b6cc2f7` |
| trench distance | `9f50256934d30187a0080b8b86ed9113cf880651bdf488ea492ef56e001531ec` | `dec158b290aef6064b1ab542d86d346c3921f96802c7764bac1833ac4501ba42` | `223cf7ed8e158906f405605713a0100ea5fcbb2e0213039ffda3ef37c1fd4a23` |
| trench side | `689b24819f5b8d8959b5d1e39c40aa2afe1029958f648e271a96b159034332c5` | `d88c15e35c21acc540c8ca208b89265d12fce42c39f8be9d63bf8ddcf0a9b938` | `f48287292e749335f72839a2ff228097ab157eebee0c46e10570e58c04bcd999` |
| trench topology | `889d7b7dbbb5ff786336b12608c72ef2e306128544dcfc5f3e4482bec11f4c52` | `f5b37754ba65fb2408df92e700644f9d150ccbf40358554e783061b8c355c927` | `d12efa398fc8edbe730e7202c4dca6daeda08e3764a66354e94520049f8e62e1` |

Every smoke gate independently reloads 92 finite model leaves and 185 finite
optimizer leaves, verifies the exact panel manifest and reward/completion
contract, and records zero mass, target, or obstacle integrity violations.
All five jobs passed the pinned four-GPU CUDA, cuDNN, NCCL, seven evaluator
tests, and two training-receipt tests before the smoke. This authorizes their
continuous production bodies; it is not yet task-learning evidence.

A proposed bitwise cross-process prefix gate in terra-baselines `d17d0be` was
tested and rejected, then removed by `7879be1`. All five update-1 aggregate
payloads match their original 500-update counterpart exactly apart from
`run_name`, but the first aggregate differences appear at update 15 and all 92
model leaves differ by update 100 in the first three runs. The maximum
per-leaf absolute difference is `0.585-0.869`. The only configuration
differences are run/path labels and the declared 500-versus-1,000 horizon;
learning rate, entropy schedule, PPO, seed, maps, reward, and model are
unchanged. Therefore cross-process bitwise equality is not a valid gate for
this GPU training path. Treatment/config receipts and deterministic
source-disjoint task curves remain authoritative; the continuous 1,000-update
runs are repeats of the same recipe, not exact replays of the first process.

Every update-1 hard integrity count is zero. The new W&B run IDs are:

| Panel | W&B run |
|---|---|
| foundation geometry | `a4vgn9wc` |
| foundation distance | `edjt0yxt` |
| trench distance | `ytprpw04` |
| trench side | `sxqzfr3t` |
| trench topology | `cqp6v20x` |

Interim continuous-boundary audit: **PASS at update 500 for all five runs**.
Each process crossed update 500 without restart and has exactly 500 aggregate
receipts plus checkpoints 100/200/300/400/500. Across all 2,500 receipts,
mass-residual, target-mutation, obstacle-mutation, and step-reward-residual
hard violation counts are zero; maximum mass residual is zero and the largest
sub-threshold floating reward residual is `4.76837158203125e-07`. This is an
integrity milestone only. The fixed-bank ten-checkpoint result remains the
learning gate.

Continuous 1,000-update training result: **PASS for all five panels**.

| Panel | Job / elapsed | Training-gate SHA-256 | FINAL SHA-256 | Online done / episodes |
|---|---|---|---|---:|
| foundation geometry | `8651897` / `01:32:19` | `d3f82192ae1ef8e2833d5b2a2887ce49faf8d5e6e2bc28faccc7be77a212d664` | `be0e6b524cbd51a7af05b5f74d73e80f96c277cc0fe6a8be1d9bc7d79bc7577b` | 54,222 / 331,247 |
| foundation distance | `8651899` / `01:33:57` | `08abaf5f1fe17c1ce597df7e7a9fd7a2e8f1a0e7047411f3c570fe05f87443d9` | `44a4014ed3399350c39d666c412697325dcbcbae270592cfee46e1fbe3da3723` | 320,758 / 537,739 |
| trench distance | `8651901` / `01:33:23` | `c234110f27df5bf363f3ad467545a10a159b96097a9237a178137995521a60d7` | `41c73cc28bf485b1d57565920660359609ac04ec5420ac486e792d4f7247bd3f` | 640,854 / 828,426 |
| trench side | `8651903` / `01:32:14` | `32b9faa94fdae1d3139234543d6a5756996f0309abca7221090462bbc2060e11` | `8c85be08ec71376ab41443b6a8c0967ed2704150cd17e9be0f24fc2a36ed4312` | 804,548 / 992,903 |
| trench topology | `8651905` / `01:32:06` | `354abb6e1e0896043f250c4562856448b7dfa3ec6cf21bc96f979a27225b1d32` | `b44d8b3894325e4f87314b3e042e5a150570d1b37414452304fc759721503329` | 81,282 / 348,385 |

Every gate certifies exactly 1,000 ordered aggregate receipts, ten numbered
checkpoints at updates 100 through 1,000, one FINAL exactly equal to update
1,000 across 92 model and 185 optimizer leaves, zero hard integrity counts,
maximum mass residual zero, and maximum sub-threshold floating reward residual
`4.76837158203125e-07`. All jobs exited `0:0`. The online counts above are
diagnostic only and do not rank or promote panels.

Continuous 1,000-update development result: **three panels continue; two stop
for diagnosis; no cell has a dynamic witness**.

| Panel | Eval job / elapsed | Evaluation SHA-256 | Best source-disjoint evidence | Decision |
|---|---|---|---|---|
| foundation geometry | `8651898` / `00:18:52` | `22e936a3e44e1d96ef8a01f1b3095fbcfb507c0c2b0028b1f8909996ab5dff0d` | one OSM success at 500; OSM median `0.878 @500`; procedural `0.852 @700`; both collapse by 800–1,000 | continue to 2,000 |
| foundation distance | `8651900` / `00:18:55` | `1ad753e03ee71fac00cac6c7d726c80d3f911ce8a396bd6b11eddae88ee1667a` | one d06 success at 300, d02 at 600, d08 at 700; all cells zero at 900/1,000 | continue to 2,000 |
| trench distance | `8651902` / `00:18:57` | `87b0a3b5efdd8cb88ec98598fc66f16937e12f9bfb1228057c8a1edb753868b1` | no successes; best medians d02 `0.582 @200`, d04 `0.624 @300`, d06 `0.573 @300`, d08 `0.213 @400`; all cells zero from 600 | stop and diagnose |
| trench side | `8651904` / `00:18:28` | `247595d095f353de97bf6a9db48bbdce2e50ae9c69d06cd323c7faafb4bd059a` | no successes; both-side `0.428 @200`, one-side `0.305 @200`; both zero from 500 | stop and diagnose |
| trench topology | `8651906` / `00:19:09` | `b8d0d14840a59024551b8c40f0db7af82ba6f27cb0d117c83254c67c1f0fed0d` | one segmented-2 success at 400; three total successes at 900 and four at 1,000, including two straight and two segmented-2 with trajectories | continue to 2,000 |

All 50 scheduled evaluations have zero integrity failures and exact reset
verification. Geometry and foundation-distance satisfy the frozen
five-evaluation rule because their last success/improvement at 700 is only
three evaluations old, despite terminal collapse. Topology has direct success
growth at 900/1,000. Trench distance and side have at least five consecutive
flat evaluations after their last improvement and are stopped. None reaches
6/8 in any cell at two consecutive checkpoints, so B0 remains unchecked and
no 120-hour recipe is qualified.

The next compute decision is therefore asymmetric:

- launch fresh continuous 2,000-update repeats only for foundation geometry,
  foundation distance, and trench topology, preserving all earlier best
  checkpoints;
- do not spend more unchanged PPO compute on trench distance or trench side;
  diagnose their mid-run-to-zero collapse before selecting a repair; and
- keep any entropy/reward/map repair separate from the unchanged 2,000-update
  repeats.

terra-baselines `008b5bd` extends the sealed launcher and evaluator to the
2,000-update/20-checkpoint milestone. `bash -n`, ShellCheck, and whitespace
checks pass.

Continuous 2,000-update submission receipt at
`2026-07-26T11:48:51+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_panels_u2000_v1`;
- source revisions: Terra
  `6702cdfa4926b37e34f62501a21dff7f3460b905` and terra-baselines
  `008b5bdd11437777a44821dfe5886b5e9ac2d6ab`;
- source-manifest SHA-256
  `96852d44712c59ffe9ea26c43c38377239043183ccd6428a5c4926e47473bc20`;
- unchanged bank-manifest SHA-256
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`;
- creation-receipt SHA-256
  `f3419501d679ad39121a0844eef2712919eb4653c34221ca845932b92a7a5927`;
- submission-receipt SHA-256
  `11f8b2928c5465f3fdf65500c8b55b56f669fe652c224310361a7091c822cd2f`;
- source, copied bank, bank-internal manifest, exact revisions, declared panel
  subset, and 2,000-update target all passed before submission;
- every job excludes only `eu-g6-064`.

| Panel | Continuous training | Dependent evaluation |
|---|---:|---:|
| foundation geometry | `8656160` | `8656161` |
| foundation distance | `8656162` | `8656165` |
| trench topology | `8656166` | `8656167` |

Trench distance and trench side are absent by design. This is a compute
receipt only; all three new runs require fresh update-1, 2,000-receipt,
20-checkpoint, and fixed-bank gates.

Continuous 2,000-update update-1 result: **PASS for all three submitted
panels**.

| Panel | Checkpoint SHA-256 | Aggregate SHA-256 | Smoke-gate SHA-256 |
|---|---|---|---|
| foundation geometry | `a532a2f59fd8c90c72296167fda3c3c798664e579676e75b32230d66844eb658` | `041bc3ebb75c21b921eeaae1de44d9bb1d7973873e7adc3a97b4113800449521` | `4e1cd162962c0ff00061e065ea9da505c9a928b78a6b4002b7d915b792f62f54` |
| foundation distance | `a0dfd18de4dcd8fc41baf064caa4fbd9b8f61a036b8171b54b3627295c1bcfa4` | `7e0c97d7809879bf88fa4228726173a4f24871d71eea8860393919476777ddbf` | `9ffde7d5ec61a3d872280da7fa7713e4abcfff156ea66d806cf338797952e1e6` |
| trench topology | `e4fdb0304b8d951ec6183d937705016c2e7dece4d63f1149fe78016f4beb20b5` | `efa3c50d99ccb7ea6c4c3b60cb340328df8e0c7bbcbee74fd95b6798c0bb1c1c` | `ed84a5bfacfc40a8110d38b01b5352f034bbca10579460b3174096e3d268ee35` |

Each gate reloads 92 finite model leaves and 185 finite optimizer leaves,
matches the exact train-panel manifest and declared reward/completion contract,
and records zero mass-residual, target-mutation, or obstacle-mutation
violations. All three production bodies started after this gate. This is an
integrity authorization only; the 20-checkpoint source-disjoint task curves
remain the learning decision.

The corresponding continuous-run W&B IDs are foundation geometry
`6fbwzje9`, foundation distance `ncy5e6yo`, and trench topology `mma0bakl`.
They are operational pointers only; W&B online aggregates do not determine
continuation or promotion.

Continuous 2,000-update training result: **PASS for all three submitted
panels**.

| Panel | Job / elapsed | Training-gate SHA-256 | FINAL SHA-256 |
|---|---:|---|---|
| foundation geometry | `8656160` / `02:56:39` | `397510e7ac9242e78bfb0daab841e822a50dae497f8b278c0994b99b80d0334b` | `4e258f041186e6961dd2f8118d7850b735a68c8ad822bde98c3d056055660a74` |
| foundation distance | `8656162` / `03:26:15` | `6afa85ae40b728d4528fac547608b8b87573b6a12ac499cf0d31a149e2f618e7` | `d11122ebfa6c7dcd856aafaf98c4a10cb16e2037e9f462bdce151cd945a8cb4e` |
| trench topology | `8656166` / `02:58:07` | `949e6d8db5e91d40c41c7784c9f0eff6237f09b883e8e5f5304101af25ec26d8` | `49d1a5d2141dcefbc38193a2223076d36ea7337467ecb9b24b4a6d2e4c71531b` |

Each gate has exactly 2,000 ordered aggregate receipts and 20 numbered
checkpoints, and FINAL exactly matches update 2,000 across 92 model and 185
optimizer leaves. The hard mass, target, obstacle, and per-transition reward
counts are zero; maximum mass residual is zero and maximum sub-threshold
step-reward residual is `4.76837158203125e-07`. This is terminal training
integrity, not a learning pass.

The three completed fixed-development decisions are deliberately
asymmetric:

| Panel | Eval job / elapsed | Evaluation SHA-256 | Source-disjoint result | Decision |
|---|---:|---|---|---|
| foundation geometry | `8656161` / `00:32:37` | `efa70d65b2acbec058502739d0d19e2aecbbb268801051632d01a4040b760d12` | zero successes at all 20 checkpoints; procedural median nevertheless rises from its prior best `0.6632` to `0.7569 @1600` | continue once to 5,000 |
| foundation distance | `8656165` / `00:31:50` | `70466947332bb67923262d306bec55a180e1f3791c0812cd7417ef5b3c639017` | isolated 1/8 successes for d04 at 500, d02 at 600, and d06 at 1,100; every cell is 0/8 at updates 1,600-2,000 and no final-window median exceeds its prior best | stop and diagnose |
| trench topology | `8656167` / `00:32:49` | `776dbdc9b903226d57e777b669b777735a4091092037dcd3da9b2e875c844bb5` | best aggregate is 5/48 at update 1,100; final-window totals are 1, 2, 0, 3, 0 and no cell exceeds 2/8 | stop and diagnose |

All three evaluators verify exact resets and zero rollout-integrity failures.
No panel has a 6/8 two-consecutive-checkpoint cell witness, so B0 remains
unchecked. Geometry receives the single fresh continuous 5,000-update repeat
authorized by the frozen median-completion rule; it is not promoted and does
not qualify for `gpuhe.120h`. Foundation distance and trench topology receive
no more unchanged PPO compute.

Foundation distance receives exactly one evaluator-only policy cross before
any repair is selected. Its sampled online train stream ends at 2,204/2,204
completed successes at update 2,000 while greedy source-disjoint development
is 0/32 throughout updates 1,600-2,000. Because the modes differ, that gap does
not yet prove identity memorization.

The train/development × deterministic/four-sampled cross completed and passed
post-run acceptance:

- manifest-sealed diagnostic root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_foundation_distance_policy_cross_u2000_v1`;
- it reuses the exact 2,000-update bank and update-2,000 checkpoint, whose
  manifest and checkpoint SHA-256 values are
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`
  and
  `5e03e5d2b64ef95ee249e030733b9d1120f55721897e2826cb7cd1c72c8b57af`;
- diagnostic source revisions are Terra
  `16e640a2cd732b363efb358db2a4760010a78acf` and terra-baselines
  `df3defdfc95c875c0e557fcecc08eb40c9fa8e5f`. Relative to the training
  Terra revision `6702cdfa4926b37e34f62501a21dff7f3460b905`, only this document
  and unused trench-diversity builder/test files differ; no imported runtime
  source changed;
- source-manifest, creation-receipt, runtime-equivalence-receipt, and
  submission-receipt SHA-256 values are
  `968590621907ddaccf07ec3dce06a2c15ca86e9da5bf4683a48932567678ce15`,
  `95e898a3131c9a1e4ab9580a3e8d38ef5d03e156b0114b05f45d038db029d7c4`,
  `2a93910f00775226beafb5b726fd4375af4192cc70c7b18d62454b8c2088178e`,
  and
  `15fb60c612a06db35b557b2b69b3f86b0fb231bd2045a0f609a6b8ee7244b4da`;
- the runtime-equivalence receipt was written after submission but before the
  first rollout. It is retrospective pre-rollout verification, not a
  pre-submission gate;
- job `8668675` completed with exit code `0:0` in `00:16:55`; the effective
  allocation was `gpuhe.4h` with a `02:00:00` limit, despite the packaged
  header naming `gpuhe.24h`;
- the output SHA-256 is
  `b72f30ad61ba72e56cd02624287f735cefb833bcd49e1cae885f6286f92ba279`;
  all 407 source-manifest files, all 3,393 bank-manifest files, and the exact
  checkpoint were re-hashed after completion; and
- the post-run acceptance receipt SHA-256 is
  `2528d466e2ae82570b46c4a747356590ad260975b60a2f9c18fc27888256800e`.
  It verifies exactly ten records, the declared deterministic seed plus four
  sampled seeds on both splits, 32 episodes per record, exact 32-slot resets
  with `env_steps == 0`, the exact train/development manifest hashes, and zero
  integrity failures.

| Split | Deterministic | Four sampled seeds |
|---|---:|---:|
| exact training identities | 31/32 | 128/128 |
| source-disjoint development identities | 0/32 | 2/128 |

Both sampled development successes are in `f_apron_d06`; d02, d04, and d08
remain 0/32 sampled. The train-minus-development gaps are `0.96875`
deterministic and `0.984375` sampled. Action sampling therefore does not
explain the collapse: this is strong training-identity memorization with
source-disjoint generalization failure.

The only authorized repair is increased foundation source-geometry diversity
while keeping the four d02/d04/d06/d08 cells, corrected dense recipe, seed,
model, PPO settings, exact development identities, and evaluation contract
fixed. Start at a fresh continuous 1,000-update milestone. If rule 13 detects
even slight held-out improvement, run the full 2,000 and then 5,000 milestones;
once the recipe clears the twice-observed family/cell qualification, move it
to a continuous 20,000-update `gpuhe.120h` production run. An unchanged
distance 5,000-update run and reward, entropy, architecture, or cell-specialist
treatments remain unauthorized.

The foundation-distance diversity-only bank was generated and statically
accepted at `2026-07-26T16:57+02:00`:

- canonical local candidate:
  `/home/lorenzo/moleworks/.artifacts/terra_b0_foundation_distance_diversity_20260726`;
- schema `terra_b0_foundation_distance_diversity_v1`, with 64 training source
  geometries and eight unchanged development source geometries shared across
  each of `f_apron_d02`, `f_apron_d04`, `f_apron_d06`, and `f_apron_d08`;
- 288/288 unique map IDs and target arrays, 72 paired geometry groups,
  source-disjoint train/development splits, and 64/8 unique train/development
  sources. Every paired group preserves its excavation geometry while changing
  only the declared dump-distance cell;
- the exact original eight training groups and all eight development groups
  are retained. All record fields except the intentionally renamed stratum,
  and all target, occupancy, dumpability, action, and distance tensors, match
  the frozen B0a identities byte-for-byte;
- the source corpus, five procedural generator files, B0a identity and file
  manifests, and base B0 builder are all SHA-256 pinned before generation;
- an earlier 0.995-IoU candidate was rejected after detecting two
  train/development near-duplicate geometries. The accepted generator caps
  cross-group dihedral IoU at 0.95, rejected two training proposals, and
  observed a maximum of `0.9470198675496688`;
- every emitted dataset was passed after writing through Terra's exact runtime
  loader: four cell directories and one panel directory for each split, ten
  directories total. This rechecks exact visible-mask capacity, tensors,
  metadata, distances, slot enumeration, and source provenance on the bytes
  training will consume;
- identity-manifest, `provenance.json`, `validation.json`, and `files.sha256`
  SHA-256 values are
  `da0b1ee39bea8af1a85b7c2fbbdb491a38c0981c167aec264bae5198adf3fae9`,
  `0e4069dc1e648b723423dc2e44a78389122f63a05674d33c37506ef658520ab2`,
  `f48afeb33ff82ce043a7b94ae8dcb39f53176d8e555e4c04748e43190677ca84`,
  and
  `ac3232e3322b4268fbb35027f0e5734a1da33e86d40f72295af9b0730f04e940`;
- all 3,493 manifest-listed files re-hash successfully. The focused new and
  existing generator/loader suites pass (`14 passed, 4 subtests`), as do Black
  and whitespace checks; and
- visual review of train, development, and paired galleries confirmed varied
  foundation footprints, large all-around dump aprons, monotonic
  d02/d04/d06/d08 spacing on fixed paired excavations, no obstacles, and no
  obvious impossible map.

This is static bank evidence, not a learning result. It authorizes exactly one
fresh continuous 1,000-update diversity-only foundation-distance treatment.
Its development decision remains the exact unchanged 32-map bank. A single
additional success in the family/current worst cell or `0.01` absolute median
terminal-completion gain triggers the fresh 2,000-update milestone under rule
13; the same rule governs 5,000. Only two clean source-disjoint qualifications
authorize the separate 20,000-update, five-day `gpuhe.120h` production run.

The 1,000-update treatment was packaged and submitted at
`2026-07-26T17:09:10+02:00`:

- manifest-sealed root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_foundation_distance_diversity_u1000_v1`;
- source revisions are Terra
  `008cb6c76e201bf135520df196b8b8eb3b78dfe2` and terra-baselines
  `51fd8d3e4478c828860ae0a4654b6484fda41db7`;
- source, outer-bank, and internal-bank manifest SHA-256 values are
  `52f4ab57ba99769d674918ac6d97a09001eadc5c9f2f616e2c765b936e73dbb4`,
  `1e49c9a1c5fa1541026a9c5a13a864c536fbb6d09bfff242f5a7c7e83c6412a8`,
  and
  `ac3232e3322b4268fbb35027f0e5734a1da33e86d40f72295af9b0730f04e940`,
  covering 409, 3,494, and 3,493 files. All re-hash successfully, and the
  copied source and bank files are read-only;
- exact train/development panel-manifest SHA-256 values are
  `ca172b2e37a6f11baeed54560238950ad4773c8125ed0960d8cbe6b52850e665`
  and
  `c1744f5c6d7f898d6b543617bbe49ab931209816e680445703f245a656a094f9`;
- creation- and submission-receipt SHA-256 values are
  `1db31e5aa66ee94006f1d0a9bc9134317237c419757a5ffb6af83bf9ff2c3c20`
  and
  `cf21ba97fef359095216ba364f0c72ea314e532a1bdc29ca3d3921ba9121d0ea`;
- train job `8674045` requests four RTX 4090s for `08:00:00` on
  `gpuhe.24h`, excludes `eu-g6-064`, and explicitly exports
  `foundation_distance_diversity_v1`; and
- evaluation job `8674046` requests one RTX 4090 and is held by
  `afterok:8674045`. It evaluates all ten scheduled checkpoints only on the
  unchanged 32-map development bank.

This is a submission receipt, not a training result. Slurm does not expose the
submitted environment through `scontrol` on this cluster, so the mandatory
causal activation check is the update-1 smoke itself: it must report
`expected_dataset_count == 256`, the exact train-manifest hash, seed
`2026072702`, finite model/optimizer/gradient state, and zero integrity
failures before the 1,000-update body is accepted.

The update-1 smoke passed and authorized the body at
`2026-07-26T17:21:49+02:00`:

- allocation is exactly four RTX 4090s on `eu-g6-025`; source/bank rehash,
  quota, CUDA library, four-device JAX, cuDNN backward-convolution, NCCL
  all-reduce, and both focused test gates passed;
- the runtime loaded exactly 256 train maps with target tensor shape
  `(1, 256, 64, 64)`, proving the diversity variant rather than the legacy
  32-map default was active;
- the smoke gate records seed `2026072702`, corrected dense reward, base
  `resnet_spatial_8x8` MLP, no warm start/resume, 92 finite model leaves, 185
  finite optimizer leaves, and zero mass residual or target/obstacle mutation;
- checkpoint, aggregate, smoke-gate, and smoke-acceptance receipt SHA-256
  values are
  `adcba1554df5c37b641afa370f85c8a1bbf5f5cc910965eb5ef3abf9b15477b3`,
  `f42056c5af0c7ac7e9af0c8c81aa83db705ccb33ec6c6d5b8560a21454e33529`,
  `5eb1f45549b5e9388e8cb899412bd25cbb76d4a196d7629b748a5a7e0c5b3dd5`,
  and
  `8e915bde41fc549f8932a3bc95ea7255d8ab31d7316723c1a51fdc81322f3c92`.

The smoke's `309.80` steps/s includes first-graph compilation and is not
performance evidence.

The full 1,000-update treatment and fixed-development decision are now sealed:

- train job `8674045` completed with exit code `0:0` in `01:33:18`, with
  exactly 1,000 ordered aggregates and ten numbered checkpoints at updates
  100 through 1,000. `FINAL` exactly equals update 1,000 across 92 model and
  185 optimizer leaves, and all leaves are finite;
- its exact training receipt, aggregate-manifest, checkpoint-manifest,
  `FINAL`, and update-1,000 SHA-256 values are
  `9cc6551b8c1fea2d080b640c31a2d7541807713149dc7e9faf8856c913fde040`,
  `a173cfbcff13dde4c36bf081f1e7ce79d6f567e1cab73276792f214b74a84151`,
  `0047f34699c94305e36e3bd99c998eea1f83d40c5fe08c630769f7621c4fc961`,
  `0dde1d1ac206fd7a44472d8ca09418f6fc2f984a252cb9387a08b8b9c6482602`,
  and
  `fed947570c444bcc21453c64db0d966d5752dc05377bc3b959d2a4a0cb5ec9ba`;
- every hard mass, target, obstacle, nonfinite, and per-transition reward
  violation total is zero. Maximum mass residual is zero and the maximum
  sub-threshold step-reward residual is `4.76837158203125e-07`;
- evaluator job `8674046` completed with exit code `0:0` in `00:18:43`.
  It covered the exact ten checkpoints and all 320 episodes on the unchanged
  32-map development bank, with exact resets at `env_steps == 0`, frozen
  source/layer hashes, and zero integrity failures;
- the overall held-out success curve at updates 100 through 1,000 is
  `0, 0, 0, 0, 0, 0, 1, 1, 6, 6`. At update 900, d02/d04/d06/d08 successes
  are `0/3/2/1`; at update 1,000 they are `2/1/2/1`;
- compared with the old same-budget 1,000-update recipe, best
  success/median-completion values improve from `1/.4975` to `2/.9452` on
  d02, `0/.3524` to `3/.9403` on d04, `1/.3475` to `2/.8809` on d06, and
  `1/.3089` to `1/.9450` on d08. Every cell passes the frozen last-five
  slight-improvement gate, with its last improvement at update 900 or 1,000;
  and
- sealed evaluation and combined acceptance-manifest SHA-256 values are
  `2f540cc5387356b752ed986e3358a260317db83c36ca5d86120df3e9dc080f22`
  and
  `c7405ba3a2101a32e29b0501cbf25c412f2ea54d329c879a0caca1863aa45283`.

No cell yet has 6/8 successes at two adjacent checkpoints, so this is strong
source-disjoint task progress but not a B0 witness or long-run qualification.
Under rule 13 it authorizes exactly one fresh, continuous 2,000-update
replication with the same accepted bank, seed, PPO, architecture, reward,
reset, entropy, horizon, and evaluation contract. It does not authorize a
5,000-update run, B0c, F1, or `gpuhe.120h` yet.

The authorized 2,000-update replication was packaged and submitted at
`2026-07-26`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_foundation_distance_diversity_u2000_v1`;
- source is pinned to the explicit Terra archive
  `3e048c661fd39b6ec6aa7e67266b160475be8e43` and terra-baselines
  `0d86f63c85c294d428694b01c34e7782b1598231`. The 412-file source-manifest
  SHA-256 is
  `8444e68c0a8807ee13166eafeccb8c30c049c7065467ad9c92ad3e7a6b522170`;
- the accepted bank was copied byte-for-byte, not regenerated. Its outer and
  internal manifest SHA-256 values remain
  `1e49c9a1c5fa1541026a9c5a13a864c536fbb6d09bfff242f5a7c7e83c6412a8`
  and
  `ac3232e3322b4268fbb35027f0e5734a1da33e86d40f72295af9b0730f04e940`,
  and its exact 256-train/32-development panel manifests remain
  `ca172b2e37a6f11baeed54560238950ad4773c8125ed0960d8cbe6b52850e665`
  and
  `c1744f5c6d7f898d6b543617bbe49ab931209816e680445703f245a656a094f9`;
- creation, wrapper-submission, and submission-acceptance receipt SHA-256
  values are
  `2ba0bd1dc896ff0dc6eb5adb11dd886c7cd9cd3f45e308b9bd1f8bbc196c0805`,
  `a74ddf1f3a5e84beeecb92fdbdf901be2451ab610063317f9cb64f5bf6bfe7f3`,
  and
  `5e8d9a388c9697952e7e14d77e89074be4aa1b9036ea207ae77616b0e1b87a4b`;
  and
- train job `8681541` requests four RTX 4090s for `08:00:00` on
  `gpuhe.24h`, with explicit
  `foundation_distance_diversity_v1`, 2,000 updates, and bytecode writes
  disabled. Evaluator `8681542` requests one RTX 4090 and is held by
  `afterok:8681541`.

This is submission evidence only. Update 1 must independently prove the exact
256-map dataset, finite model/optimizer/gradient state, frozen treatment, and
zero integrity failures before the body is accepted.

The 2,000-update foundation-distance update-1 smoke passed. Job `8681541`
started on `eu-g6-069` with exactly four RTX 4090s; the independent gate
verified the exact 256-map manifest, seed `2026072702`,
`corrected_dense_v1`/`exact_visible_dump_v1`, base
`resnet_spatial_8x8` MLP, no resume/warm start/teacher, finite model,
optimizer, and gradient state, and zero hard integrity counters or mass
residual. Its read-only smoke-acceptance receipt SHA-256 is
`933537ccf0e757f4a850321f4c688e01c7d52dd6e84c09ddc6b70fe486cd368c`.
This admits the continuous 2,000-update body only; fixed-development learning
evidence remains pending.

The 2,000-update foundation-distance treatment is fully accepted:

- train job `8681541` and evaluator `8681542` both completed with exit code
  `0:0`. Training contains exactly 2,000 ordered aggregates and 20 numbered
  checkpoints, with `FINAL == update 2,000` across 92 model and 185 optimizer
  leaves, exact source/bank/config hashes, and zero hard integrity failures;
- training-receipt, `eval.json`, independent evaluation-validation,
  evaluation-receipt, and combined-receipt-manifest SHA-256 values are
  `8589b0e905f87e408264949a6435f4cbbe77bd79933b0f94e85e10feaf8ee63a`,
  `503b926d94ec09aa785144d5f20037c6db09df1ed780461d78aeb565392a9712`,
  `0c8451ab5c59098c6f3e350584e1c2752affc5f273778164b8cf1f376223cfda`,
  `1fb209a12805c585a1d258f686f61376f5d3ead1dbe5f753e47f281037e51047`,
  and
  `58536a680907220e88d3e5a542c52e650b0f33203dba550a6dac6d94f234895e`;
- all 640 held-out episodes use exact resets and have zero mass, target,
  obstacle, nonfinite, slot, or termination-integrity failures. Positive
  illegal spill in 32 timeout records is valid under the approved cleanup
  contract and retained as soil; exact-dump-mask integrity remains one;
- overall held-out successes at updates 100 through 2,000 are:

  ```text
  0 0 0 0 0 1 4 2 1 12 12 4 11 13 17 10 12 9 13 10
  ```

- d02/d04/d06/d08 per-cell success curves are:

  ```text
  d02  0 0 0 0 0 1 1 0 0 4 2 0 2 3 4 1 4 2 6 3
  d04  0 0 0 0 0 0 1 0 1 3 4 2 2 3 6 5 4 4 2 3
  d06  0 0 0 0 0 0 1 1 0 3 5 0 5 4 4 2 3 3 3 2
  d08  0 0 0 0 0 0 1 1 0 2 1 2 2 3 3 2 1 0 2 2
  ```

- the same-budget update-1,000 checkpoint improves from the old 6/32 to
  12/32, and all four distance cells improve over the prior recipe. The
  final-five rule passes through d02 at update 1,900: 6/8 successes, a
  two-success gain over its prior best, with a clean saved trajectory; but
- d02 reaches 6/8 only at update 1,900 and d04 only at update 1,500. No cell
  has an adjacent 6/8 pair, so the panel witness and long-run qualification
  both fail.

The sealed decision is `continue_same_panel`. Exactly one fresh continuous
5,000-update replication of the same diversity treatment is authorized under
rules 13-14. It must not resume the 2,000-update process. B0c, F1, and
`gpuhe.120h` remain unauthorized.

The authorized 5,000-update confirmation was submitted at `2026-07-27`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_foundation_distance_diversity_u5000_v1`;
- source, outer-bank, internal-bank, exact 256-train, and exact
  32-development manifest SHA-256 values remain
  `8444e68c0a8807ee13166eafeccb8c30c049c7065467ad9c92ad3e7a6b522170`,
  `1e49c9a1c5fa1541026a9c5a13a864c536fbb6d09bfff242f5a7c7e83c6412a8`,
  `ac3232e3322b4268fbb35027f0e5734a1da33e86d40f72295af9b0730f04e940`,
  `ca172b2e37a6f11baeed54560238950ad4773c8125ed0960d8cbe6b52850e665`,
  and
  `c1744f5c6d7f898d6b543617bbe49ab931209816e680445703f245a656a094f9`;
- recursive source/bank equivalence, read-only sealing, exact loader counts,
  focused tests, shell syntax, and ShellCheck passed. Creation,
  wrapper-submission, and submitted-jobs receipt SHA-256 values are
  `0bc6ba6e8a34d567e4e24c052ba323fa3c379a34a5e4554d37ee2b6ee9c13ee8`,
  `b908567d29326c811562a8d34248a77951276356d7f97bad3a1870804342db5f`,
  and
  `29826e51670ae57154dd43ac787677227d855a6e1095c85dedc5330b4e767ec9`;
  and
- train job `8727052` requests four RTX 4090s for `16:00:00` on
  `gpuhe.24h`; evaluator `8727053` requests one RTX 4090 under
  `afterok:8727052`. The run is fresh from seed `2026072702`, with no resume,
  warm start, or teacher, and covers 655,360,000 environment transitions.

This is submission evidence only. Runtime GPU/update-1 acceptance remains
mandatory before the 5,000-update body is admitted.

The 5,000-update runtime smoke passed on `eu-g6-034`:

- exactly four RTX 4090 devices passed JAX CUDA, cuDNN backward convolution,
  NCCL `pmap` all-reduce, and in-job test gates;
- the runtime loaded the exact 256-map treatment and produced finite model,
  optimizer, gradient, and rollout state with zero mass residual, target or
  obstacle mutation, and per-transition reward violations;
- seed, reward, model, PPO, reset, horizon, and no-resume/no-warm/no-teacher
  configuration match the accepted treatment; and
- smoke-gate, final-smoke-checkpoint, update-1 aggregate, and read-only
  smoke-acceptance receipt SHA-256 values are
  `897721e1beadfd934afcece5e4fff6e73408fcf33a818e11796bde969e02ee3f`,
  `3292008bf89ec5ad08b38c525a922ec35b8cb36744d74c123c37214ec582e5f5`,
  `e34c609c16c14d66eb82af76e3be427e9f0ce5f7031b476880fa173aef62765a`,
  and
  `331e1f24927936a16d2d38c7934f69cc02a681955c8fee0ced18f76b880e1db5`.

This admits only the fresh continuous 5,000-update body; the 50-checkpoint
fixed-development curve remains authoritative.

Future B0 submissions use terra-baselines
`588ee4585f8348b5d52c4ee6af6c2a0261b405d9` to make treatment provenance
explicit. The submission wrapper now validates the declared
`base_v1`, `trench_side_diversity_v1`, or
`foundation_distance_diversity_v1` treatment against the requested panel,
exports the treatment to both the train and dependent evaluator jobs, records
it in the submission receipt, and requires `PYTHONDONTWRITEBYTECODE=1` so a
read-only packaged source tree cannot be mutated by imports. Focused wrapper
tests exercise the successful export/receipt/dependency path and both
pre-submission rejection paths (`3 passed`); the complete CPU suite remains
green (`137 passed`). This is a launch-provenance hardening change, not a
training treatment. It applies to every fresh 2,000-, 5,000-, or qualified
production package. The active 1,000-update package remains pinned to
terra-baselines `51fd8d3e4478c828860ae0a4654b6484fda41db7`: its accepted
update-1 smoke independently proves that the intended 256-map diversity
treatment was active, so the wrapper hardening does not invalidate that run.

Trench topology likewise receives exactly one evaluator-only policy cross at
the post-hoc update-1,100 development peak before a repair is chosen. It is a
diagnostic for action mode versus identity generalization, not an independent
held-out estimate:

- manifest-sealed diagnostic root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_trench_topology_policy_cross_u2000_v1`;
- the package uses the exact training Terra revision
  `6702cdfa4926b37e34f62501a21dff7f3460b905` and diagnostic-only
  terra-baselines revision
  `8498873457361fd78631589f9979508e7c9ba7ea`;
- source-manifest, creation-receipt, and pre-submission runtime-equivalence
  receipt SHA-256 values are
  `0cd8e87ad058af0feb581f5cd26f06a88a6b15fc794df66a75d8c373cfa8a3be`,
  `5b269a7fc4bd2f1d33710f55d1eb62e44fd30db5461e965233cd7b32e9f26e35`,
  and
  `c6b5a6985550e2bad8fc945f8da032d666f0960550b8e4dedf34975fd3e50cc6`;
- all 405 source files are hash-verified and filesystem read-only. The exact
  outer bank manifest and update-1,100 checkpoint SHA-256 values are
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`
  and
  `06082a862e6e3cfa5bd61cf925df5109c8cb4976e57e226626587b690b7a7bed`;
- exact train/development topology manifest SHA-256 values are
  `b3da1246e7529394ab9118aca03785e1d7e0616bf867e70b18c401aed22cbeaa`
  and
  `24d511ab78a7d098edce3256b160fe7bf3b11337f38ed93968f77ccb898e47e8`;
  each contains six cells times eight identities, with no map/source overlap;
  and
- job `8671422` was submitted at `2026-07-26T16:26:35+02:00` on
  `gpuhe.4h` for `02:00:00`, one RTX 4090, with `eu-g6-064` excluded and
  bytecode writes disabled. Submission-receipt SHA-256 is
  `bfba8bbf7486070d31ecd2ab8a83d6ab17e601672efb26e26db72979f64c91e2`.

Acceptance requires Slurm `COMPLETED/0:0`, the exact outer hashes, exactly ten
48-episode records, all six cells with eight episodes per record, the declared
deterministic and four sampled seeds on both splits, exact 48-slot resets with
`env_steps == 0`, and zero integrity failures. The four sampled evaluations
are stochastic replications on the same 48 identities, not 192 independent
maps. Because the development bank selected update 1,100 and was already
observed across 20 checkpoints, any repair chosen from this result must be
tested on a fresh source-disjoint bank. No topology PPO repair, specialist,
B0c, or long run is authorized before acceptance and interpretation.

The topology policy cross passed post-run acceptance at
`2026-07-26T16:58:45+02:00`:

- job `8671422` completed with exit code `0:0` in `00:20:51` on one RTX 4090;
- output SHA-256 is
  `9df0a0acaf5974dd6fec8d7b35d195570466f3f6b343d35062422a6ed1497c16`;
- all 405 source and 3,393 bank files, the exact update-1,100 checkpoint, both
  split manifests, ten 48-episode records, 480 episode records, and all 33
  acceptance checks re-verified. Exact resets passed with `env_steps == 0`;
  mass residual, target/obstacle mutation, nonfinite state, slot/termination
  disagreement, and dump-mask integrity failures are all zero; and
- acceptance-receipt SHA-256 is
  `48215735e6dc0cbcec824ee0fe281a137fd05f15fa94c0fc8ff082737d9d375f`.

| Split | Deterministic | Four sampled seeds |
|---|---:|---:|
| exact training identities | 26/48 | 107/192 |
| source-disjoint development identities | 5/48 | 31/192 |

Sampling adds only `0.015625` train success and `0.057292` development success,
while train-minus-development gaps remain `0.4375` deterministic and `0.395833`
sampled. Development T, X, and disconnected cells remain zero under all four
sampled seeds; only segmented and straight cells show any development success.
The primary failure is therefore source-disjoint topology/geometry
generalization, not greedy action selection. The post-hoc update-1,100 peak is
diagnostic only and does not qualify a recipe. It rejects unchanged topology
compute and restricts any later repair to increased training geometry or an
explicit topology ladder evaluated on a fresh source-disjoint bank.

The minimal topology repair is now preregistered but deliberately waits for
the active trench-side 2,000-update decision. A no-write feasibility dry run
can generate 64 train plus eight fresh development sources for every topology
cell under centered-dihedral IoU `<0.95`, so a full six-cell bank is
technically feasible. It is not the next scientific treatment:

- the historical B0a train/development topology splits contain cross-split
  near-duplicates despite disjoint source IDs (maximum IoU `0.959` for
  straight and `0.969` for disconnected), so those development identities
  cannot adjudicate the repair;
- T, X, disconnected, segmented, and straight cells differ materially in dig
  volume and generator structure, so a 384-map equal-count mixture would
  confound topology difficulty with the geometry-diversity treatment; and
- the active straight/side diversity run is the prerequisite causal test. If
  it plateaus with the same train/development gap, broadening topology would
  repeat an unresolved failure.

If trench-side passes or remains integrity-clean and improving, build
`B0-DIVERSITY-T-SEGMENTS-v1`:

- only `t_segmented2_both_d02` and `t_segmented3_both_d02`;
- 64 unique train sources per cell, retaining the original eight B0a train
  identities byte-for-byte and adding 56, for 128 train maps total;
- eight brand-new development sources per cell in a new seed namespace, for
  16 development maps total. No previously observed topology development
  identity may be reused;
- scratch seed `2026072705`, base `resnet_spatial_8x8`,
  `corrected_dense_v1_trench_absolute_off`, broad both-side d02 dumping,
  `3.25x` capacity, no obstacles, exact reset/evaluator, PPO, entropy, and
  horizon unchanged; and
- centered-dihedral IoU `<0.95` for every added train geometry against
  retained train and all historical development geometries, and for every new
  development geometry against all train, historical development, and earlier
  new-development candidates. Pin B0a, base builder, source corpus, and all
  generator hashes; recompute raster topology/component metadata; require
  unique source/map/dig/target hashes, at least 40% capacity on each side,
  exact loader checks, manifest, galleries, and visual review.

Its update-1 gate must prove exactly 128 train maps and the usual finite/mass
integrity contract. Evaluate the 16 fresh development maps every 100 updates.
Each cell requires 6/8 at two adjacent checkpoints plus a saved legal
trajectory. Apply rule 13 at 1,000, 2,000, and 5,000; one passing cell and one
plateau permits at most one failed-cell isolate. A still-improving but
unqualified 5,000-update screen is explicitly non-saturated and receives a
separately declared next bounded milestone, not `gpuhe.120h`. T, X, and
disconnected cells remain later K0 stages. The full 384-train/48-fresh-dev
six-cell bank is a technically viable fallback, not an authorized launch.

The geometry-only 5,000-update repeat was manifest-sealed and submitted at
`2026-07-26T15:27:11+02:00`:

- manifest-sealed root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_foundation_geometry_u5000_v1`;
- source revisions are Terra
  `146919ffd78242f1bcf6d17091a47a20fd22b2bc` and terra-baselines
  `c418bb8bc8b57102a3d6982d85c7a3c9bd6bd85a`; the latter changes only the
  bounded launcher to admit the preregistered 5,000-update milestone and a
  16-hour train request;
- the exact prior B0a bank was copied without regeneration. Source-manifest,
  bank-manifest, creation-receipt, and submission-receipt SHA-256 values are
  `efcffedab1beb19220c96ff626548331b272b44362a9bc42e2dcdf8972ecf9a3`,
  `98fa41ca879f3523f846e42405daa065bbbe97cfb2aab9f75bdd2288d47ffa2b`,
  `256a49d6f86c2952eed172a4bc18d6b7ada1ea3216fe93a17c285218731b789c`,
  and
  `0fceddc7941d308facb594bfa16a640ddd04c31681b065a204dfdf6eb260417d`;
- all 407 source and 3,393 bank files, revisions, scripts, syntax, 16 train
  identities, and 16 development identities passed before submission;
- train job `8667019` requests four RTX 4090 GPUs for `16:00:00` on
  `gpuhe.24h`, excludes only `eu-g6-064`, and runs continuously from scratch;
  and
- evaluation job `8667022` is held by `afterok:8667019` and requires all 50
  scheduled checkpoints.

This remains a compute receipt. Its own update-1 smoke, complete 5,000-update
integrity gate, and deterministic development curve are mandatory; it cannot
inherit evidence from the 2,000-update process.

The geometry-only repeat independently passed its update-1 smoke:

- checkpoint, aggregate, exact 16-map train manifest, and smoke-gate SHA-256
  values are
  `620e4586f720351652cb2430ec169408fd99f4dcd7239d6edb4d6d19425321a2`,
  `8f5b2fbd3afa2f3e09e88fc2d2689d85d255b9a41dc9d9708a007de5aaa65917`,
  `dd30a0e5d66ac6b66da2b4a1172d74b1a9d7a420829157c02e75a0d5beb14463`,
  and
  `ec61cebf89f33a44fea4b9e84611121aaaf3ae3e38406685cd1b5ba0d8c98199`;
- all 92 model and 185 optimizer leaves are finite and the declared seed,
  reward, architecture, entropy, PPO, reset, horizon, and 5,000-update target
  match; and
- maximum mass residual, target mutation, and obstacle mutation are zero.

This authorizes the continuous 5,000-update body. It remains an integrity
result only. The corresponding continuous-run W&B ID is `m0n2yngc`; it is an
operational pointer, not task evidence.

The geometry-only 5,000-update treatment is fully accepted, but it does not
solve either held-out cell:

- train job `8667019` completed with exit code `0:0` in `07:11:27`; evaluator
  `8667022` completed with exit code `0:0` in `01:11:56`;
- the run has exactly 5,000 ordered aggregates and 50 numbered checkpoints,
  `FINAL == update 5,000` across all 92 model and 185 optimizer leaves, zero
  hard integrity failures, maximum mass residual zero, and maximum
  sub-threshold step-reward residual `4.76837158203125e-07`;
- training-gate, checkpoint-manifest, aggregate-manifest, training-receipt,
  `eval.json`, and evaluation-receipt SHA-256 values are
  `e8410e9eaf63db1dadd9a49bef997207f59376712932e8a4e2fc8a7a389b3f5c`,
  `1e1f8d1f268c60381f2be3c6efaa5bb50aa72b54f993c022b0333c1bc8ddc37b`,
  `2b905c4795b8b3cbb4e6032d1dad7c87ace422fa5d5e47a7fa8052e774206437`,
  `3c5ef5ef0c4363f0755d252c4bc5dc15b5a2d510f9da95634ac21e6bad391752`,
  `3f76b5f1596498c8b57e8898e7b483ffffa808915a7bf78ba8a64c303f04435d`,
  and
  `815d1bcb09a9589f275d2b8b04aeb7eeaacc491201ac04dd619ea5cc2601f37f`;
- all 800 held-out rollouts use exact 16/16 resets at `env_steps == 0` and
  have zero integrity failures, but both OSM and procedural success curves
  remain 0/8 at all 50 checkpoints. There is no successful held-out
  trajectory and no cell witness;
- OSM has no final-window progress. Procedural median completion improves from
  its prior in-run best `0.8057184815406799` at update 2,400 to
  `0.8233191668987274` at update 4,800, a preregistered gain of
  `0.017600685358047485`; and
- the frozen decision is therefore `continue_same_panel`, solely through the
  procedural completion event. It is not a success, B0c promotion, family
  qualification, or long-run authorization.

This result weakens the belief that unchanged low-diversity geometry training
is close to solved. Any same-recipe continuation must be declared as a new
bounded milestone, remain separate from the diversity treatment, and cannot
use `gpuhe.120h` until an actual repeated held-out witness exists.

The trusted C5 population receipts materially narrow the stopped-trench
interpretation:

| Panel | Sampled online train success at 900 | Sampled online train success at 1,000 | Deterministic source-disjoint development |
|---|---:|---:|---:|
| trench distance | 2,218/2,227 | 599/808 | 0 successes from 600 through 1,000 |
| trench side | 2,528/2,554 | 1,519/1,630 | 0 successes from 500 through 1,000 |

The update-900/update-1,000 aggregate SHA-256 pairs are respectively
`7bc262988bec5aea7c29a7229277c398182ec048a12d6132fab9bf41e2a9d12f` /
`7a12ada87ae206666fe300ebdb3499c4e9308da1566a8e3ba8ce26e77b5d8ddd`
and
`bf4fbb3c6a1b14a3435d5e948c66fb119b1f51ae0ad4a28e337645202839e9c1` /
`84405d0a1556420272a51f7df39abff9c0fb9a34a70176ddee65a5ed3d05043f`.
Therefore the development result is not evidence that the sampled policy
globally stopped moving dirt. It leaves two crossed explanations: train versus
source-disjoint identity generalization and sampled versus deterministic
action selection. Both must be measured on the same fixed checkpoint and
reset bank before selecting an entropy, reward, or map repair.

The stopped trench panels have a separate, bounded action-trace diagnosis
queued at `2026-07-26T11:56:10+02:00`. This is evaluator-only compute: it
does not add PPO updates or alter the fixed bank.

- immutable evaluator root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_collapse_diag_v1`;
- source revisions: Terra
  `9ca7bceebde5d84700290faa3a805183bb281b24` and terra-baselines
  `8a3f510e1422e8feaf73965e55886d2c1f9b0fd7`;
- source-manifest SHA-256
  `66774e156e9e640465fdb8196e15e8ec4268e3ce1583b6cf065a09373aadc8f9`;
- creation-receipt SHA-256
  `c88e2d3c023a76a93025fab499d08258198d808fd236194af33ae3dfeada76c3`;
- submission-receipt SHA-256
  `b57f4eead653b246be8a44351eafc92070d083aa25ff4ed8714a6f977de04885`;
- initial jobs `8656488` and `8656489` were cancelled at `00:05:55` after
  prematurely interpreting the compile-averaged first-rollout throughput as
  steady-state throughput. The reference evaluator shows the first replay
  rising from `0.1` to `1.3` steps/s as JIT compilation amortizes, followed by
  approximately `5.5` steps/s;
- source and treatment remain unchanged. Replacement receipt at
  `2026-07-26T12:03:12+02:00` has SHA-256
  `bfab28ce0125fd74d9ae3d705c37b7423c7e65de704e880a392223ca0619a6d1`;
- replacement trench-distance job `8656748` replays updates 200, 300, 600,
  and 1,000 over all 32 panel maps;
- replacement trench-side job `8656750` replays updates 200, 400, 600, and
  1,000 over all 16 panel maps; and
- each replay records the eight action-mode counts, effective-action counts,
  first/effective dig step, switches, run-length traces, terminal dig/dump
  mass, completion, and exact-reset verification. Both jobs exclude
  `eu-g6-064`.

Both replacements passed: distance completed in `00:11:01`, side in
`00:11:45`, with exit code `0:0`, exact-reset verification, and zero rollout
integrity failures. Their output SHA-256 values are respectively
`f86fbdc3a64a22a241d71f4c41e10bc53215614cff7524317b8d625edd107dc2`
and
`3bf209cef67fe6db659890cd853153f3b4dae04addfe5a9aa81ee7aa277925f7`.

| Panel / update | Maps issuing `DO` | `DO` / effective `DO` actions | Median action switches out of 449 | Deterministic development result |
|---|---:|---:|---:|---|
| distance / 200 | 30/32 | 993 / 177 | 438 | cell medians 0.157-0.582 |
| distance / 300 | 26/32 | 488 / 148 | 284 | cell medians 0.195-0.624 |
| distance / 600 | 4/32 | 163 / 26 | 444 | every cell median 0 |
| distance / 1,000 | 4/32 | 14 / 14 | 444 | every cell median 0 |
| side / 200 | 14/16 | 322 / 53 | 223 | both-side 0.428; one-side 0.305 |
| side / 400 | 9/16 | 318 / 36 | 442 | both-side 0.421; one-side 0 |
| side / 600 | 1/16 | 2 / 2 | 444 | both cells median 0 |
| side / 1,000 | 2/16 | 217 / 8 | 442 | both cells median 0 |

The deterministic development failure is thus an action-mode change, not a
global physics veto: all four late distance maps that issue `DO` have at least
one effective `DO`, and all 14 update-1,000 distance `DO` actions are
effective. Side also retains at least one effective `DO` path. The greedy
policy instead spends nearly every transition alternating forward, backward,
and rotation actions; the high switch count rules out a single static no-op
but identifies movement chattering. Side update 1,000 repeats many ineffective
`DO` actions on only two maps, so it has both chattering and local repeated-dig
behavior. These traces explain how the visible zero is produced; they do not
yet distinguish whether sampled action selection recovers useful behavior or
whether that behavior generalizes.

The output is diagnostic evidence for selecting a single controlled repair,
not a promotion gate by itself. After the action-trace replay, run the minimal
fixed-checkpoint train/development by deterministic/sampled cross above.
In particular, an entropy, reward, or map change must not be inferred solely
from the already observed deterministic development zero.

That crossed evaluator was implemented and queued at
`2026-07-26T12:10:52+02:00`:

- immutable evaluator root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_policy_cross_v1`;
- source revisions: Terra
  `caad73962b5ab8a916a6919ec2926f074010c4ae` and terra-baselines
  `34fcbb0d7352d1ef03019b27e3a912f8bde98662`;
- source-manifest SHA-256
  `c12b73adf6000effa70f6d3e6148fd72c2e1faaac5c881f6ecab06c310eeb44e`;
- creation-receipt SHA-256
  `cefaf3382ad5ff87e0141f9de67e8bbce71e35730743a6dac4dcae5f8696fb1c`;
- submission-receipt SHA-256
  `acefbb2a2c185948d97d449af773769bffce8f0047bc3bef199998285bf4f1ad`;
- trench-distance job `8657124` and trench-side job `8657126` each evaluate
  only their immutable update-900 checkpoint;
- each job crosses exact train and source-disjoint development identities with
  one deterministic replay and sampled seeds 2026072801-2026072804; and
- the two pure aggregation tests, byte compilation, Black, bash syntax,
  ShellCheck, source receipt, exact reset, and zero-integrity gates are
  mandatory. Both jobs exclude `eu-g6-064`.

This is an evaluator-only causal diagnosis. It does not make update 900 a
promotable checkpoint and does not authorize a training repair until its cross
is adjudicated together with the action traces.

Both crossed evaluators passed their exact-reset and zero-integrity gates.
Distance job `8657124` completed in `00:22:51`; side job `8657126` completed
in `00:18:55`; both exited `0:0`. Their output SHA-256 values are
`3a41bc64fc79cc73bb3077f34c0691e0be064583ec21745bbd026a27515d86b6`
and
`4560d16b8342c310a284a94f0aa3815b92e42ac51ecbb4db7aaf08ecfb9ebf56`.

| Panel | Train deterministic | Train sampled, four seeds | Development deterministic | Development sampled, four seeds |
|---|---:|---:|---:|---:|
| trench distance | 28/32 | 111/128 | 0/32 | 15/128 |
| trench side | 15/16 | 61/64 | 0/16 | 0/64 |

The distance sampled-development successes decrease with dump distance:
6/32 at d02, 5/32 at d04, 3/32 at d06, and 1/32 at d08. In contrast, each
distance train cell has 7/8 deterministic successes, and sampled train success
is 27-28/32 per cell. Side has 7/8 deterministic train successes for
both-side, 8/8 for one-side, sampled train rates 29/32 and 32/32, and zero
sampled development successes in both cells.

This adjudicates the main failure as source-identity memorization:

- deterministic train-minus-development gaps are 87.5 percentage points for
  distance and 93.75 points for side;
- sampled train-minus-development gaps remain 75.0 and 95.3125 points;
- side has no sampled-versus-deterministic development advantage, so an
  entropy/action-selection repair is not authorized;
- distance sampling recovers only 11.71875% development success and degrades
  monotonically with distance. This is useful policy-mode evidence, but not a
  reason to move far dumping into the starter curriculum.

The single next trench training treatment is therefore
`B0-DIVERSITY-T-SIDE`, not another unchanged run and not an entropy, reward,
or architecture ablation:

- keep exactly `t_straight_both_d02` and `t_straight_one_d02`;
- expand only training diversity from eight to 64 unique geometries per cell;
- preserve the original eight train identities as an exact subset and use the
  exact same eight source-disjoint development identities per cell;
- keep scratch seed 2026072704, `corrected_dense_v1_trench_absolute_off`,
  PPO, base `resnet_spatial_8x8`, entropy schedule, reset, and 450-step horizon
  unchanged;
- start one continuous 1,000-update run, evaluate every 100 updates, require
  6/8 per cell at two consecutive checkpoints with a saved successful
  trajectory and zero integrity failures, and apply the global slight-progress
  rule for a 2,000/5,000 extension; and
- if this diversity-only treatment still memorizes, stop before changing
  reward or architecture and inspect its train/development trajectories.

The four-cell distance panel receives no new PPO allocation now. Its d02
behavior is covered by the close side treatment; d04-d08 remain ordered later
curriculum cells, with far dumping admitted only after the close cell
generalizes.

The diversity-only bank was generated and statically accepted at
`2026-07-26T12:40+02:00`:

- local immutable candidate:
  `/home/lorenzo/moleworks/.artifacts/terra_b0_trench_side_diversity_20260726`;
- schema `terra_b0_trench_side_diversity_v1`, with 64 training identities and
  eight development identities for each of `t_straight_both_d02` and
  `t_straight_one_d02`;
- 144/144 unique map IDs and target arrays, source-disjoint train/development
  splits, preserved paired dig geometry, and all static capacity, obstacle,
  source, and loader checks passed;
- the original first eight training identities and all eight development
  identities per cell match the frozen B0a bank in all 18 declared identity,
  geometry, target, and validation fields. The 32-row reference gate used B0a
  identity-manifest SHA-256
  `911b6e3a453d6d9e1aeaebfe5fcef33406c89aae0180e1c4eb8739efc1fd5b4e`;
- 31 training and one development proposals were rejected as templated
  duplicates before the declared counts were reached;
- `files.sha256`, `provenance.json`, `validation.json`, and the paired panel
  gallery have SHA-256 values
  `3e9059d7e167f8b0f054c46a9da2b3b5f1e1d6991041f57fde4e4ca3f953f4eb`,
  `4658f7611e5590cb34051bb3802ba5dc4269098157df47568262a6d242976bff`,
  `177740f1b3f9e04ce9f21c56cb467fffca3fd4ceec5fe71f283d6e8018c8a91c`,
  and
  `a6cfbe8dd7a25d1931dbaf6834a31482262d7b6ae3f721fad407001f014f9a1b`;
- visual review of the train and development galleries confirmed close,
  oversized side-cast regions, distinct both/one-side access, varied trench
  position/orientation/length, no obstacles, and no obvious impossible map;
  and
- the new generator tests plus the original B0 feasibility-panel tests pass
  (`6 passed`), as do byte compilation, Black, and whitespace checks.

This is a static bank receipt, not a learning result. It authorizes exactly
one 1,000-update `B0-DIVERSITY-T-SIDE` run after the bank and source are copied
and hash-verified under a new immutable Euler root.

That treatment was packaged and submitted at `2026-07-26T12:50+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_trench_side_diversity_v1`;
- source revisions: Terra
  `db6bead7ee9360bb6a203c8b58f0956984bc73d6` and terra-baselines
  `3dcb7d2400c391f0630cc8d2a05ce408761187ab`;
- source-manifest, copied-bank-manifest, creation-receipt, and submission-receipt
  SHA-256 values
  `7eefb6a986a9ed0beda8e80140817321fa7f6b88bdbd85d6627b8ec36a919d9b`,
  `3711c0d6c3ff715afcbf296926ba6061dfe5d172c496d5af285b4e2929a30ace`,
  `5805ec5d83b98c6da59b1ce8dd6a9522c3251ddbb7b4b7c4db95fcced15f1a31`,
  and
  `2e695fcca670538964b256e08f618f993144e5bc6db060790667a0d9dab7e73d`;
- source, all 1,753 copied bank files, the bank-internal manifest, exact
  revisions, 128-train/16-development counts, validation status, script syntax,
  and all outer hashes passed before submission;
- training job `8658605` runs one continuous from-scratch 1,000-update
  `trench_side_diversity_v1` treatment on four RTX 4090 GPUs, with seed
  2026072704 and only `eu-g6-064` excluded; and
- fixed-development evaluation job `8658606` is held by an `afterok`
  dependency on the complete training and terminal integrity gate.

This remains a submission receipt, not evidence of learning. Update 1 must
pass the exact checkpoint/optimizer/aggregate/dataset gate before the
production body is accepted; the 10-checkpoint source-disjoint curve then
decides pass, bounded continuation, or stop.

The update-1 smoke gate subsequently passed:

- checkpoint, aggregate, 128-map train manifest, and smoke-gate SHA-256 values
  are
  `9721979acde7700327b6d6012337d8896b0572ad7ee3e8fd980e3f828da538b5`,
  `a56b0c5701cc9989e1f3821a759d1bf29ceb7f9ff93d3a3421cb3d08ed265fdc`,
  `183444a6be146b097afe56199abd9a83eda71e8dd9a5d7c1d72f3f2573c75dab`,
  and
  `a6bd892801ffea9c6cbd397ba36bab48a548aab037a892a82dbaf3564dbcb6e9`;
- all 92 model leaves and 185 optimizer leaves are finite;
- the exact seed, reward, architecture, entropy, PPO, reset, 450-step horizon,
  and one-update checkpoint configuration match; and
- maximum mass residual, target mutation, and obstacle mutation are zero.

This authorizes the continuous 1,000-update body. It remains an integrity
result only, not a map-learning or promotion result.

The corresponding continuous-run W&B ID is `aknko0s4`. It is an operational
pointer only; its online aggregates do not decide continuation or promotion.

The continuous 1,000-update training result is **PASS**:

- job `8658605` completed in `01:47:52` with exit code `0:0`;
- training-gate, FINAL, last-update, numbered-checkpoint-manifest, and
  aggregate-manifest SHA-256 values are
  `9360afde2c8204f6fd77b45983efaad582344a213c49373ab0391c51c2cf515e`,
  `3295b55872a7ec0e1a10cff648dccfd749dda7e42e594fb2483adfbc6ac06893`,
  `c28907b4e7b95ee95af0794cf8b743874b766136ea93b9a7e7fe8768d23ab411`,
  `d25daf3bc60866a34e913d34acdf2ad429719ad0c279b65f6474c1297df28ba8`,
  and
  `dcb7e4d75d4881d3dfe227c72647ce9b250404530a0137107663c8d22331aeb4`;
- all 1,000 ordered aggregate receipts and all 10 numbered checkpoints at
  updates 100 through 1,000 are present; FINAL exactly equals update 1,000
  across 92 model and 185 optimizer leaves;
- the exact 128-map manifest is revalidated; all mass, target, obstacle, and
  per-transition reward hard violation counts are zero; maximum mass residual
  is zero and maximum sub-threshold step-reward residual is
  `4.76837158203125e-07`;
- 199 completed episodes exceed the informational independently accumulated
  episode-return drift tolerance. This is the explicitly non-blocking float32
  association metric documented by the C5 schema-v2 amendment; the
  transition-level reconstruction hard gate is zero; and
- the online stream contains 120,380 task completions out of 376,932 completed
  episodes. That sampled-train quantity is diagnostic only and is not evidence
  of source-disjoint generalization.

The dependent fixed-development result is **CONTINUE SAME PANEL**, not a B0
witness:

- job `8658606` completed in `00:19:05` with exit code `0:0`; its sealed
  `eval.json` has SHA-256
  `fb2fe771a387540306a27c85e14fbbfd278a2fda1b0ddea9b26966b13082d92d`;
- exact-reset verification passed for all 16 development identities at
  `env_steps == 0`, with zero integrity failures;
- the held-out success curve at updates 100 through 1,000 was:

  ```text
  update                         100 200 300 400 500 600 700 800 900 1000
  t_straight_both_d02 successes    0   0   0   0   0   1   1   1   2    3
  t_straight_one_d02 successes     0   0   0   0   0   0   0   1   2    4
  ```

- at update 1,000 the median absolute completions were 0.4044 for both-side
  and 0.5000 for one-side, and a successful action trajectory was saved for
  each cell;
- neither cell achieved 6/8 at even one checkpoint, so neither has the
  required two-consecutive-checkpoint witness and B0 remains unchecked; but
- both cells improved inside the final five-evaluation window, including
  success gains at update 1,000. The preregistered slight-progress rule
  therefore authorizes exactly one fresh, continuous 2,000-update replication
  of the same diversity-only treatment. It does not authorize B0c, a reward or
  architecture change, a 5,000-update run yet, or a 120-hour production run.

Because checkpoint-v2 does not serialize the environment, RNG, and action
history required for a bit-exact continuation, the 2,000-update treatment must
start from scratch and run continuously. Its source, bank, seed, PPO,
architecture, reward, reset, entropy, horizon, and development gate remain
unchanged. A later 5,000-update extension is allowed only if its final
five-evaluation window again satisfies the same held-out slight-progress rule;
the 120-hour queue remains reserved for a recipe that clears the repeated
family/cell gates defined below.

That bounded replication was sealed and submitted at
`2026-07-26T15:14:31+02:00`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_trench_side_diversity_u2000_v1`;
- the exact prior source and bank were copied, rather than regenerated:
  Terra `db6bead7ee9360bb6a203c8b58f0956984bc73d6`, terra-baselines
  `3dcb7d2400c391f0630cc8d2a05ce408761187ab`, 128 train identities, and
  16 source-disjoint development identities;
- all 2,160 source/bank files passed hash verification. Source-manifest,
  bank-manifest, creation-receipt, and submission-receipt SHA-256 values are
  `7eefb6a986a9ed0beda8e80140817321fa7f6b88bdbd85d6627b8ec36a919d9b`,
  `3711c0d6c3ff715afcbf296926ba6061dfe5d172c496d5af285b4e2929a30ace`,
  `bf6149f7be1450ba74fd4df2306f43c782c58754f544e5d607dd21131f23f7ed`,
  and
  `157b136e786d66abe05b9e50411e43ad00c236767bb1cf91d4358dbe795dab33`;
- train job `8666365` requests four RTX 4090 GPUs on `gpuhe.24h`, excludes
  only `eu-g6-064`, and runs 2,000 continuous from-scratch updates with the
  unchanged `trench_side_diversity_v1` treatment; and
- fixed-development job `8666366` has an `afterok:8666365` dependency and
  applies the same 20-checkpoint witness/continuation adjudication.

This is a submission receipt only. Update 1 must pass the exact smoke gate
before the production body is accepted, and the source-disjoint development
curve—not online sampled-train success—decides the next allocation.

The 2,000-update diversity replication independently passed its update-1
smoke:

- checkpoint, aggregate, unchanged 128-map dataset manifest, and smoke-gate
  SHA-256 values are
  `0c4eb6d10b8f0da7121afebe0201d03ec8320d6e2760ded234f88babaff47f21`,
  `4a2b5a60aa35f25ee17c768a70cd94441da07b16ad55a2aa40772b69302ad6cc`,
  `183444a6be146b097afe56199abd9a83eda71e8dd9a5d7c1d72f3f2573c75dab`,
  and
  `437cce0eeb8881f354b9a6a821918b22bb2aecd6b33bee41b7e17aa8a72f2b5e`;
- all 92 model and 185 optimizer leaves are finite and the exact declared
  seed, reward, architecture, entropy, PPO, reset, and horizon match; and
- maximum mass residual, target mutation, and obstacle mutation are zero.

This authorizes only the continuous 2,000-update body. It is an integrity
result, not a trench-side witness. The corresponding continuous-run W&B ID is
`d5srft2n`; it is an operational pointer only.

The 2,000-update training and fixed-development decision are now sealed:

- train job `8666365` completed with exit code `0:0` in `02:55:20`.
  It contains exactly 2,000 ordered aggregates and 20 numbered checkpoints at
  updates 100 through 2,000; `FINAL` exactly equals update 2,000 across all 92
  model and 185 optimizer leaves, and every saved leaf is finite;
- all source and bank files re-hash, and all hard mass, target, obstacle,
  nonfinite, and per-transition reward-violation totals are zero. The maximum
  mass residual is zero and the largest sub-threshold step-reward residual is
  `4.76837158203125e-07`;
- the post-training acceptance receipt at
  `manifests/post_training_acceptance_receipt.txt` has SHA-256
  `c80b40551b2a638967790108cc4237cc2d7a8cd703484ea5d00e7b25d57140f8`;
- evaluator job `8666366` completed with exit code `0:0` in `00:32:53`.
  Its 20 checkpoints, 320 episodes, exact 16-slot resets at
  `env_steps == 0`, checkpoint/config hashes, terminations, trajectories, and
  integrity fields all pass. The sealed `eval.json` SHA-256 is
  `c910e59ed4392cae94a91917d5c0fd75f08115db188ef56e35ac1076b09c3a61`;
- held-out successes at updates 100 through 2,000 are:

  ```text
  both-side  0 0 0 0 0 1 1 0 2 4 6 6 4 3 6 7 6 7 5 2
  one-side   0 0 0 0 0 1 0 1 3 5 3 4 4 4 3 3 3 4 3 3
  ```

- both-side now has accepted consecutive witness pairs at 1,100-1,200,
  1,500-1,600, 1,600-1,700, and 1,700-1,800, with saved legal trajectories;
  one-side never reaches 6/8. Its final-five success curve is
  `3, 3, 4, 3, 3` against its prior best 5, and its final-five median
  completion remains below its prior best 1.0; and
- the frozen evaluator and an independent decision reimplementation both emit
  `conditional_cell_isolates`. The evaluation acceptance receipt at
  `manifests/evaluation_acceptance_receipt.txt` has SHA-256
  `16274bc40e0de2eb588c96fc440b032ae15b9aea818975d7b21d71c3ca91729a`.

The literal family total improves from a prior best 10/16 to 11/16 at update
1,800, but that extra success is entirely in the already witnessed both-side
cell. The more specific B0b rule controls: compute is allocated to an unpassed
cell, and a mixed pass/fail panel permits at most one same-budget conditional
single-cell isolate. Therefore this result authorizes exactly one fresh,
continuous 2,000-update `t_straight_one_d02` isolate with the same map
identities, seed, reward, PPO, architecture, reset, horizon, checkpoint, and
fixed-development contract. It does **not** authorize a full-panel 5,000 run,
B0c, F1, or a 120-hour production run. If the isolate fails the same
two-consecutive 6/8 witness gate, one-side is dynamically unproven and this
recipe stops before any reward or architecture change.

The conditional isolate was implemented at terra-baselines
`0d86f63c85c294d428694b01c34e7782b1598231` as one explicit
`trench_one_d02_isolate` path rather than an ambient variant:

- it reuses the accepted first-class
  `cells/train/t_straight_one_d02` and
  `cells/development/t_straight_one_d02` views already present in the sealed
  diversity bank: exactly 64 train and eight development identities, with
  manifest SHA-256 values
  `00f1e32c15c11a7013f09a22047f1ca3c8dc1f6cb706dc6f8fa3faa8119c4020`
  and
  `804fea45a2a87f3abfe4df835de70dfbd1e92d9d472a936c86a3fdf3f715c29d`;
- train and evaluation launchers hard-require exactly 2,000 updates, the
  explicit single panel, 64/8 dataset counts, the original seed, and the
  original corrected trench treatment. The isolate is absent from the default
  multi-panel submission list;
- the checkpoint verifier freezes the exact cell paths and all PPO, reward,
  reset, horizon, architecture, and seed values. A failed isolate still
  records slight-progress diagnostics but can emit only
  `stop_and_diagnose_panel`, never another continuation authorization; a true
  adjacent 6/8 witness can still pass normally;
- direct runtime validation loaded exactly the 64/8 single-cell views with
  no map/source overlap, and an independent review found no P0/P1 defect;
  focused evaluator/submission/training-receipt tests pass (`18 passed`), the
  complete CPU suite passes (`143 passed`), and Black, byte-compilation,
  `bash -n`, ShellCheck, and whitespace checks are clean; and
- packaging must copy the entire accepted 1,753-file bank byte-for-byte and
  preserve outer bank-manifest SHA-256
  `3711c0d6c3ff715afcbf296926ba6061dfe5d172c496d5af285b4e2929a30ace`.
  A regenerated or hand-filtered bank is rejected.

The one-shot isolate was packaged and submitted at `2026-07-26`:

- immutable root:
  `/cluster/scratch/lterenzi/codex_terra_edge_runs/curriculum_recovery_v1_20260725/b0_trench_one_d02_isolate_u2000_v1`;
- source is pinned to Terra
  `3e048c661fd39b6ec6aa7e67266b160475be8e43` and terra-baselines
  `0d86f63c85c294d428694b01c34e7782b1598231`. The 410-file source-manifest
  SHA-256 is
  `85f2582af3a8b028deadc7192d1044fc62637bae7203feaab3dba2666cdc4dcb`;
- all 1,753 outer-bank files and all 1,752 internal-bank files re-hash
  successfully. Exact-loader validation confirms 64/8 unique one-side maps,
  64x64 shape, capacity ratio 3.0, and no train/development map or source
  overlap. Its loader-receipt SHA-256 is
  `e7933f9f960807c53b5f0d471a09de68f80132b91454cd6ae4b5a5fe9382c635`;
- creation, wrapper-submission, and submission-acceptance receipt SHA-256
  values are
  `fd331acb2b0efabf0f9666e710bcc4b5bfc595d976b8f7867231a874203cc680`,
  `deff6cecc88ce20e03e17f2db36d010039ff35679d724b3cbc594c6c5b6ae29a`,
  and
  `84722b11d69d9ca7f4c4f8fde86c86e8a309da1cf40cf9bc614e2499c5697a30`;
  and
- train job `8681252` requests four RTX 4090s for `08:00:00` on
  `gpuhe.24h`; evaluator `8681256` requests one RTX 4090 under
  `afterok:8681252`. The train allocation started on `eu-g6-071` with exactly
  four verified RTX 4090s.

This remains submission/infrastructure evidence. The exact 64-map update-1
smoke must pass before the one-shot 2,000-update body is admitted.

The isolate update-1 smoke passed:

- an independent validator reloaded the exact 64-map
  `t_straight_one_d02` manifest, verified the preserved source and bank
  hashes, and proved numbered/final equality across all 92 model and 185
  optimizer leaves;
- every leaf and gradient diagnostic is finite, with gradient norm
  `0.165367946`, and mass, target, obstacle, and per-transition reward hard
  violations are all zero; and
- independent-validator and read-only smoke-acceptance receipt SHA-256 values
  are
  `680e87e5cb68083bdd6892dec13422fc2f74198c063565e361a30c6acf2d7b57`
  and
  `92232dca83108bad516688b11466b53867699003df5774515c62e43b2b57ef3b`.

This admits the continuous 2,000-update body only. The fixed eight-map
development evaluation remains the one-shot learning decision.

The one-shot isolate completed and its sealed decision is
`stop_and_diagnose_panel`:

- train job `8681252` completed with exit code `0:0` in `03:07:13`, with
  exactly 2,000 aggregates and 20 checkpoints, exact
  `FINAL == update 2,000`, finite model/optimizer state, intact 410-file
  source and 1,753/1,752-file bank manifests, and zero hard integrity
  failures;
- terminal-training validator and read-only training-receipt SHA-256 values
  are
  `c21f9f2b4a10654ae4510ce7fd13c6e3ab3a01ef529d4c28321509a622c96ad1`
  and
  `3658b35f050081afa6cb82720146bcd842e5a6862dc2980ff0c970bfb20bcf8a`;
- evaluator `8681256` completed with exit code `0:0` in `00:31:54`. All 160
  episodes terminate, exact resets and slot coverage pass, and mass, target,
  obstacle, nonfinite, termination, and reward-integrity failures are zero;
- held-out successes at updates 100 through 2,000 are:

  ```text
  0 0 0 0 0 0 2 3 3 3 4 3 3 2 2 3 1 1 2 2
  ```

- the best checkpoint is only 4/8 at update 1,100; the final-five curve is
  `3, 1, 1, 2, 2`. There is no 6/8 checkpoint, adjacent witness, or
  final-window slight improvement. Fourteen clean successful trajectories
  prove partial feasibility, not robust generalization; and
- independent evaluator-validation and read-only evaluation-receipt SHA-256
  values are
  `784c4d548c0d163ff745f0a33d9ddaa1595541fda7076f58e8b945ca9925e8f0`
  and
  `2a90bd01a36585fd7592d6dcd6db09a8cf41349abfd3eae259968345deb51881`.

The roughly 96% late sampled-training success against 25% final held-out
success is direct overfitting evidence. Single-cell specialization did not
repair source generalization, and performance regressed after the update-1,100
held-out peak. The one permitted isolate is exhausted: no same-treatment
continuation, full-panel 5,000 run, segmented-trench bank, B0c, F1, or
`gpuhe.120h` run is authorized. Before choosing a new treatment, audit
train/development geometry support and best-versus-final action trajectories;
do not infer a reward or architecture change from this result alone.

The combined post-run belief update is:

- corrected task semantics, exact dump masks, spill cleanup, mass
  conservation, and numerical stability are working; none explains the
  remaining failures;
- increasing source geometry diversity materially improves
  foundation-distance transfer, including d06/d08, so far dumping is not
  currently the primary blocker;
- fixed-bank performance is highly non-monotonic. Promotion-bank checkpoint
  selection and a separate untouched development bank are required before
  long training;
- one-side trench maps are feasible but the current generator/training
  support does not generalize robustly; and
- unchanged 16-map foundation-geometry training is much weaker than the
  256-map distance-diversity treatment. More compute is justified only as a
  bounded preregistered continuation, not as proof that the recipe is ready.

The read-only geometry-support audit narrows the next diagnostics:

- foundation OSM/procedural train and development maps have the same easy
  all-around dump regime, no obstacles, median dump distance one tile,
  reachable-workspace coverage at least 0.9957, and capacity at least 20.56x.
  Capacity, access, and dump distance do not explain zero held-out success;
- the procedural split has a real eight-map geometry-support gap: train angles
  are concentrated at 0/30/60/75/105/165 degrees, while development adds
  45/90/120/135 degrees and higher aspect ratios. OSM development shapes are
  moderately larger but overlap train more closely;
- during updates 4,501-5,000, sampled training success is 60.88% OSM and
  12.29% procedural, versus 0/400 fixed-development successes per cell. Every
  development identity nevertheless reaches at least 0.897 completion at some
  checkpoint, then coverage switches identities and regresses; and
- failed action sequences were not persisted, so the accepted artifacts
  cannot yet separate identity memorization from sampled action selection or
  terminal cycling.

Before another foundation-geometry PPO run, authorize one evaluator-only
policy cross at updates 3,800, 4,800, and 5,000 on exact train16/development16
maps, deterministic plus four frozen sampled-action seeds. It must produce 30
records/480 rollouts, exact resets, all integrity fields, and failed
action/effect/completion traces for selected high-completion and zero-progress
identities. If every mode remains train-high/development-low, the next PPO
treatment is geometry diversity only (at least 64 train identities per cell,
balanced procedural angle/aspect and matched OSM volume). Sampled
development successes instead select an action-selection diagnosis; neither
train nor development success despite near completion selects a
terminal-cleanup/cycle diagnosis. A longer unchanged geometry run waits for
this cheaper cross.

The trench support audit likewise rejects a longer unchanged isolate:

- all eight development maps lie inside the 64-map training ranges for coarse
  trench geometry, orientation, dump distance, capacity, and margins. They
  share balanced sides, close dumping, no obstacles, and no disconnected
  regions;
- development success frequency still tracks local raster support:
  Spearman rho is `-0.875` versus nearest-train raw-mask Hamming distance and
  `+0.850` versus maximum dump-mask IoU (`n=8`, descriptive only). Three maps
  never solve, while better-covered maps dominate successes;
- best update 1,100 and final update 2,000 trajectories are legal and
  effective, several failures reach 0.892-0.989 completion, and retained
  checkpoints work. Reward integrity, primitive execution, and checkpoint
  loss are therefore not the first explanations; and
- the prior paired both/one-side run peaks at 5/8 on the exact one-side bank,
  versus 4/8 for the isolate. Doubling isolated one-side exposure did not
  help; paired both-side examples appear mildly regularizing.

No trench PPO continuation is currently authorized. The smallest candidate
new treatment, after a train-only static bank audit and explicit authorization,
is the prior paired both/one-side recipe with training diversity alone raised
from 64 to 256 independent maps per cell. Keep the close-straight generator,
reward, architecture, PPO, reset, horizon, and evaluation cadence fixed.
Generate without consulting development identities, retain the current bank
only for diagnosis, and require a new untouched source-disjoint promotion/test
bank before any future qualification. Reward, entropy, and architecture
ablations remain deferred.

#### B0c — Expand only witnessed easy cells

> **Superseded proposal.** The 64-identities-per-cell B0c bank below was never
> the v0.3 release contract. The live path is S1 support audit, S2 448-scenario
> review, S3 witnesses, then expansion of only selected active cells to
> 256 independent training identities per cell.

The primary easy bank required by F1 is:

```text
foundation:
  f_osm_all, f_procedural_all, f_apron_d02, f_apron_d04

trench:
  t_straight_both_d02, t_straight_one_d02,
  t_segmented2_both_d02, t_segmented3_both_d02
```

Each family unlocks B0c and its own F1 independently once all four of that
family's easy cells pass B0b. Foundation must not wait for trench, or vice
versa; this makes the B0c/F1 boundary consistent with rules 15-16. A combined
generalist bank still waits for all eight cells. For each unlocked family,
regenerate its four cells from the same frozen algorithm, but with disjoint
identities:

- 64 unique training identities per cell;
- eight promotion identities per cell;
- eight development identities per cell; and
- eight sealed identities per cell.

The resulting foundation and trench family banks each contain 256 training
maps and 32 maps in each evaluation split; the later combined easy bank
contains 512 training maps and 64 maps in each evaluation split. Use a new
seed/map-ID namespace: no source, map, target, or identity from B0a or any
diversity, repair, or diagnostic panel may appear in the expanded bank, and
the four expanded splits must be mutually source-disjoint. The current B0a
and trench-side-diversity builders deliberately preserve panel
prefixes/identities and therefore cannot be reused unchanged.

B0 is complete only after the expanded-bank exact-count loader,
split/source/hash disjointness, C1a capacity, numeric-range, paired-geometry,
visual, memory, and exact production-shaped update-1 compile gates all pass.
F1 qualification uses only the frozen promotion split, asserts exactly 32
family and eight per-cell episodes, and requires adjacent scheduled
100-update evaluations; development remains diagnostic and sealed remains
unopened. The distance-6/8, T/X, and disconnected cells remain named K0
candidates even when they pass B0b; they are not mixed into F1.

Before materializing or launching the fixed bank, measure loader memory and
first-update compile with the intended 64x64 arrays. The training contract is
512 unique maps total: 64 identities for each of eight cells. If that fixed
shape does not fit, stop and revise the bank contract explicitly; never
silently reduce diversity after launch.

## 12. Phase K — global quantitative map curriculum

### K0 — Execute the accepted v0.3 admission plan

The normative map definitions, eight-condition pilot, accepted reviewer
comments, and open choices now live in
[`MAP_BENCHMARK_SPEC.md`](MAP_BENCHMARK_SPEC.md). The older OSM-to-procedural
and d02-to-d08 linear ladder is superseded: source is separately gated
provenance, and d02-d08 measures separation rather than loaded transport.

Current implementation checklist:

- [x] S0 records the accepted three-depth design and append-only reviewer
  decision log.
- [x] The active
  [`DIGGING_BENCHMARK_SITE_GOAL.md`](DIGGING_BENCHMARK_SITE_GOAL.md)
  standardizes current B0a design-input records, exports map graphics and
  distributions, supports data-backed human review, and serves a local
  inspector plus organized image folders before the admitted S2 bank is
  ready. Production deployment is deferred by user choice.
  Execution receipt: the non-admission preview contains `256` scenarios,
  `16` legacy cells, `144` source groups, `1,792` verified layer PNGs,
  `16` overview sheets, `256` individual composites, and zero sealed assets.
  Exact source/output hashes, verification commands, responsive screenshots,
  local URL, implementation commit, and unresolved Static/Witness/S2
  limitations are recorded in `DIGGING_BENCHMARK_SITE_GOAL.md`. No Lorenzo
  decision or S2 admission is claimed. Site commit `91ccf87` additionally
  exposes the verified `64`-map/`32`-pair capacity bank as the default review
  dataset, keeps B0a selectable, and passes comment-before-decision
  persistence plus exact-hash JSONL round-trip on live port `4173`. Site
  commit `21d5526` adds a separate `16`-map narrow large-foundation bank,
  retains both earlier releases byte-identically, and explicitly labels the
  unpaired work-size slice as a review candidate rather than a curriculum
  level. Site commit `8340fbb` makes the B0a Anchor, One-axis, and Composed
  review groups visible in their intended order with exact counts and
  one-click filtering, while explicitly stating that they are display groups
  rather than admitted levels. Its exact source/export hashes, tests, and unresolved
  Static/Witness/metre-field limitations are recorded in the site goal.
- [x] S1 derives `0.571428571428125` m/tile, the live `7 x 11` footprint,
  radial envelope, exact runtime cone masks, and protocol hash from Terra
  rather than duplicated constants.
- [ ] S1 revalidates unchanged B0 rasters, rewrites metre/static receipts,
  preserves identities that pass, and lists/replaces only failures in one
  hashed per-identity migration receipt. The old unreceipted probe count is
  not a target. Affordable live geometry, capacity, identity, split, and
  state migration now pass deterministically for all `256`; exact
  action-reachable Static fields remain blocked by the cost gate.
- [x] S1 adds explicit batched complete-agent-state reset plus admissibility
  validation and hashes every reset-consumed `Agent`/`AgentState` field.
- [x] S1 implements action-reachable exact dig-to-dump direct-service fields
  for the initial scenario before any map is described as forced rehandling;
  terminal/during-trace access fields wait for canonical witness replay, and
  relay-hop scoring is deferred until its graph is specified.
- [x] S1 runs the fixed one-identity non-admission direct-service cost probe,
  separating logical attempts from padded kernel execution and emitting no
  subset feasibility result. It passed the frozen 60-minute p95 and
  20%-memory-headroom gate.
- [x] S1 confirms the probe estimate on one complete exact 64 x 64 scenario
  before any 256-identity profile.
- [ ] S1 profiles exact direct-service validation on all 256 frozen B0a
  identities, receipts cold/steady runtime, replay counts, peak memory, and
  projected 448-scenario cost, and reviews that receipt before S2. The
  confirmation must agree within a factor of two and calibrate to at most
  24/48 hours p95 for 256/448 sequential scenarios before this profile runs.
  The first confirmation agrees at `1.0324x` but fails both time limits, so
  this item is blocked on exact-path optimization and a new staged receipt.
  The pure-GPU ladder was rejected because graph/prefilter populations differ
  by device, and the CPU-graph/GPU-service hybrid was rejected because its
  full-population service outputs and final result differ. The next bounded
  treatment may change only external scenario-level CPU process concurrency;
  validator semantics, scenario identity, and per-scenario outputs stay
  unchanged. Heading-vectorization remains unauthorized.
- [x] S1 freezes full condition IDs: source, achieved separation, named
  capacity metric/band, and pair-specific train-only audited numeric
  volume/compactness support. No S2 record retains `vmatch`. Commit
  `847322e5` materializes the eight-condition canonical registry and verifier:
  `4` foundation plus `4` trench, `3` Anchor plus `5` One-axis, literal
  prerequisites, equal weights, exact panel memberships, and `448` projected
  scenarios. The verifier re-hashes all `435` files across the three support
  trees. `maximum_centered_dihedral_iou` thresholds remain `UNSET` and
  explicitly block S2; scenario materialization, Static, Witness, and PPO
  flags remain false.
- [x] S1 Foundation retunes procedural all-around generation toward fixed OSM
  support and freezes one source-pair interval/tolerance; the apron capacity
  pair instead shares exact OSM dig identities and volumes.
- [x] S1 Trench keeps straight fixed, audits candidate segmented-2
  `U[10,13)` and segmented-3 `U[7,9.5)` length ranges, freezes only a
  well-supported common band, and receipts lengths, turns, conditioning, and
  rejection histograms. Pilot trench identities are fresh.
- [ ] S1 Capacity builds and visually verifies one exact-dig,
  separation-matched `slcap03_04`/`slcap07_10` apron pair. The fresh
  `32`-pair artifact and internal visual/integrity checks pass; exact Static
  validation and Lorenzo's recorded review decision remain.
- [x] S1 updates the history aggregator to the integer retention gates and
  focused `n=8`/`n=32`/invalid-evaluation/streak-reset tests.
- [ ] S2 builds the eight-condition 448-scenario train/promotion/development/
  sealed pilot and local review site.
- [ ] S2 records Lorenzo's per-condition visual decisions before expanding an
  active training cell to 256 identities.
- [ ] S2 foundation review reports target-area and required-volume coverage
  explicitly and includes a deliberately larger-footprint candidate slice.
  Small foundations remain valid anchors, but the current B0a source-bank
  preview is not evidence of broad work-volume coverage. Admission as a
  work-volume condition still requires train-only numeric support and a
  450-step witness.
  - [x] A train-only visual candidate now covers `328-339` cells
    (`8.01-8.28%` of the site) with all-around dumping, no obstacles, and
    `11.08-11.49x` single-layer capacity. Its deterministic full-tree rebuild
    and exact-loader checks pass. This closes the requested larger-footprint
    visual slice only: it is a narrow generator tail, is not source-matched to
    OSM, does not establish broad `8-12%` support, and remains Static/witness
    pending and non-admitted.
  - [x] The corrected v2 artifact regenerates all 48 p50/p95/max metre fields
    from the frozen live `36.5714285714 / 64 =
    0.571428571428125` m/tile receipt instead of stale `0.6875`. All 16
    identities satisfy exact tiles-times-tile-size equality; a disagreeing
    benchmark/environment protocol now fails closed. Deterministic rebuild and
    exact-loader verification pass. Compared with v1, only
    `identities.jsonl`, `provenance.json`, `summary.json`, and
    `files.sha256` change; every NPY and PNG is byte-identical, and identities
    are byte-identical after deleting the three corrected metre fields.
    Artifact
    `/home/lorenzo/moleworks/.artifacts/terra_pilot_large_foundation_review_20260727_v2/`
    has manifest SHA-256
    `3def81e558ff9776bdb0e9c8e17969d0c2c91f6ee43d98e80b2d227cc01a79b0`.
    This repairs metadata only and still grants no Static, witness, broad
    support, or admission claim.
- [ ] S3 supplies exact replay witnesses within the 450-step protocol and
  reports witness margin; any stricter publication-Core cutoff requires an
  evidence-backed spec revision.

Admission remains deliberately small:

- each condition declares literal direct prerequisites;
- sibling one-axis conditions are independently witnessed rather than joined
  in a stage-wide AND;
- generalist optimizer lineages are still cumulative and sequential; sibling
  policies are never merged;
- promotion requires `6/8` per new/direct-parent cell twice, and a four-cell
  family panel additionally requires `26/32` twice;
- OSM/procedural source cells cannot hide one another;
- each mastery reference is the lower count from its two passing evaluations;
- each prior 8-map condition retains at `max(6, reference-1)`, and the fixed
  32-map family panel retains at `max(26, reference-1)`; and
- two consecutive complete integrity-valid retention failures stop the run,
  restore the last passing checkpoint, and relaunch the previous mixture as a
  new recorded treatment. A pass resets the streak; invalid/incomplete
  evaluations block promotion but do not diagnose policy regression.

Generalists first sample foundation/trench 50/50. Within a family that has an
active frontier, the first rehearsal treatment samples 50% frontier and 50%
uniformly over that family's admitted cells; a family without a frontier
samples its admitted cells uniformly. Specialists apply the same rule inside
their one family. Two retention failures invoke rollback. No ambiguous
recent-cell bucket or learned/adaptive scheduler is authorized without a new
named causal treatment.

Every map-curriculum run uses one immutable materialized map level, 100% full
resets, horizon 450, DENSE rewards, `apply_trench_rewards=false`, and the
frozen corrected reward/action/observation/dynamics hashes. Development never
drives promotion; sealed evaluation opens once after model selection.

## 13. Conditional architecture work

### A0 — Keep role-specific base and medium models until a representation failure exists

Use base `resnet_spatial_8x8` for F0-F1-G0 and medium
`resnet_spatial_8x8_se` after S0. This is a lineage plan, not an architecture
ablation. No v5, 128-resolution, transformer-core, or recurrent sweep is
authorized.

Authorize `resnet_spatial_8x8_se` versus
`resnet_spatial_8x8_se_xattn` only if:

- C0-C4 and C1a pass;
- F0, F1, G0, and S0 pass;
- K0 fails specifically on location-conditioned constrained maps;
- train-identity learning is strong;
- observation alias tests are resolved; and
- trajectory/error analysis shows incorrect selection among spatially
  separated legal work or dump regions.

The ablation must use the same corrected observation, map bank, PPO settings,
parent-policy decision, and budget. Primary metrics remain source-disjoint
success and retention, not old-map SWHiR.

Recurrence is considered only after a consequential alias cannot be represented
by a compact explicit state feature.

## 14. Separate progressive reward curriculum

### W0 — Correct the reward specification before implementation

Revise `PROGRESSIVE_REWARD_CURRICULUM.md` so:

- Stage 1 parity is defined against the newly frozen corrected dense contract;
- task completion and termination share C1's single source of truth;
- legacy inconsistent dense behavior remains replayable but is not the v2
  parent;
- map level cannot change reward stage;
- terminal duplication/backfill is disabled for the single-agent v2 path; and
- reward-independent success, workspace-cycle, and step metrics decide.

Apply the same corrected Stage-1 parent and gate language to
`PROGRESSIVE_REWARD_VALIDATION_PLAN.md`; the two specifications must not define
different qualification contracts.

Do not implement a compatibility framework beyond the one legacy replay path
needed to evaluate existing checkpoints.

### W0a — Keep the dense transport ablation explicit and conditional

`corrected_dense_v1` is the only authorized F0 reward. A second dense reward is
not silently folded into C1/C1a.

Candidate `transport_potential_v1`:

```text
cost(state) =
    sum(off-zone positive soil volume * distance_to_accepted_dump)
  + carried_soil_volume * agent_distance_to_accepted_dump

potential(state) = -cost(state)
shaping = gamma * potential(next_state) - potential(state)
```

If implemented, this term replaces the current dump-time relocation-potential
reward; it is not added on top. Loading transfers the same soil mass from world
cost to carried-load cost at the same location, a legal dump drives that
mass's cost to zero, and a wrong dump leaves a positive off-zone cost. The
potential never changes action validity.

Before implementation, ratify:

- traversable shortest-path versus Euclidean distance;
- how carried-load distance is defined when the base cannot reach the dump
  mask directly; and
- normalization/capping without breaking the potential telescoping property.

Trigger a dense A/B only if `corrected_dense_v1` passes F0 but shows a
transport-specific failure on the first otherwise-feasible constrained cell,
or Lorenzo explicitly authorizes it. Use the same scratch initialization,
maps, reset seeds, PPO, 100-update evaluation cadence, and global
slight-improvement continuation rule. Do not combine this with a map-stage,
architecture, terminal-reward, or partial-reset change.

### W1 — Qualify one foundation dense parent

Dependency: S0 passes and one fixed corrected foundation family passes its
family/cell gates in three scheduled evaluations.

Freeze:

- map family and bank;
- full resets;
- model and PPO;
- corrected reward contract;
- checkpoint hash; and
- evaluation protocol.

Only this qualified parent authorizes the reward experiment.

### W2 — Run the matched reward experiment

Arms:

```text
A: corrected dense continuation
B: corrected dense -> terminal objective
C: corrected dense -> terminal margin -> terminal objective
```

Keep map and reset distributions identical. Select lexicographically by:

1. fixed-bank success;
2. productive workspace cycles on episodes solved by both;
3. steps on episodes solved by both; and
4. completion margin on failures.

Validate the selected sequence separately on trenches and only then on the
multitask mixture. Do not combine it with a new map stage.

## 15. Separate partial-reset curriculum

### PR0 — Test partial resets after the map sampler is selected

Compare:

- control: 100% untouched full resets;
- treatment: 75% full resets and 25% mass-conserving partial resets.

Start the partial share with equal 50% and 75% `in_zone` states. Add 25% states
only as a bridge if needed; defer `mixed`, `near_zone`, and 90% states.

Hold map, reward, model, PPO, and budget fixed. Primary evaluation remains
untouched full tasks; report partial-reset success separately by completion
fraction and pile mode.

Support the treatment only if it improves late-state competence without
reducing full-task family/cell success or retention.

## 16. Later map expansion

Admit one axis at a time after the applicable K0 quantitative cells pass:

1. disconnected structural foundations and multi-junction trenches;
2. stronger single obstacles;
3. combined road/wall/object sites;
4. increased dig/dump separation at matched capacity and direct-service
   coverage;
5. reduced direct-service coverage and finally witnessed forced rehandling;
6. tight natural capacity as its own matched axis;
7. physical boundary-spill dynamics; and
8. the final realistic deployment mixture.

For a separation or rehandling axis, create an exact-geometry pair that holds
capacity and site access fixed. A passed local/direct-service parent is a
prerequisite. Separation labels come from achieved validator bands;
forced-rehandling requires zero initial direct-service coverage plus a
450-step completion witness.

Unsolved or statically valid but dynamically unproven families remain in a
named challenge bank; they are not mixed into training and called curriculum
difficulty.

## 17. Immediate execution queue

This is the live top-level checklist. A box is checked only after the
acceptance evidence in the corresponding section passes.

1. [x] Freeze D0 and reject both completed historical curriculum arms.
2. [x] Ratify and implement C0-C4 plus the contained transition C1a.
3. [x] Close the C1b excavator-footprint integrity defect before training.
4. [x] Complete C5 auditable population aggregates and exact finite-checkpoint
   terminal smoke.
5. [x] Freeze the two independent F0 launch/evaluation paths, exact
   hyperparameters, checkpoint-lineage gates, and reduced-shape PPO integration
   smoke at terra-baselines `6c56525`; correct the receipt gate without changing
   the treatment at `c58ad23`.
6. [x] Finish D1/D2, inspect every JSON integrity field, and write the
   preregistered materiality/memorization/policy-mode decisions.
7. [x] Run independent update-1 finite GPU smokes for the foundation and trench
   F0 jobs, reload each exact saved checkpoint, and verify the C5 receipt.
   The corrected v2 smokes passed in retry jobs `8632268`/`8632271`; their
   independent hashes and configuration receipts are recorded in F0.
8. [x] Launch the two scratch F0 fixed-identity probes with
   `corrected_dense_v1`; evaluate 32 fixed seeds every 100 updates. The first
   attempt is preserved as failed/cancelled infrastructure evidence. Clean
   replacements from immutable root `f0_retry1` completed: foundation passed
   feasibility but failed terminal retention, while trench failed cleanly.
9. [x] If either F0 arm fails, stop its descendants and run only the
   trajectory/O0/transition/reward diagnosis implicated by that arm. The exact
   replay gate passed and selected F0R without authorizing an observation or
   architecture change.
10. [x] Implement and run F0R. The shaping-off trench treatment passes at
    updates 300/400 and retains a terminal 900/1,000 pair; together with the
    retained foundation update-900 witness, this authorizes B0 to build and
    validate the orthogonal feasibility panels and admit only dynamically
    witnessed cells.
11. [ ] Complete S1: canonical state/factor IDs, live-geometry revalidation,
    migration/cost receipts, action-reachable direct-service validation,
    pair-specific foundation/trench support audits and generator retunes,
    integer retention gates, and the exact-dig
    `slcap03_04`/`slcap07_10` apron generator.
12. [ ] Complete S2: build the balanced four-foundation/four-trench
    448-scenario bank and record Lorenzo's visual decisions in the local site.
13. [ ] Complete S3: replay exact, mass-conserving witnesses within 450 steps;
    keep unsupported maps in the named Challenge rather than training on them.
14. [ ] Expand only selected witnessed active cells to 256 independent
    training identities per cell, then run each family specialist with the
    source-disjoint promotion and retention gates. A qualified family earns
    its fresh 20,000-update `gpuhe.120h` run without waiting for the other.
15. [ ] If both specialists pass, run the 50/50-family generalist; only a
    twice-qualified generalist becomes the new-distribution small teacher and
    may start the admission graph one isolated axis at a time.
16. [ ] Execute W1/W2 and PR0 only as separate treatments after map-sampler
    selection. Open the sealed bank once after all model/treatment selection
    and publish the final causal, integrity, compute, and checkpoint receipts.

R0 is not in the default launch queue. Authorize it only if D2 confirms
train-and-development regression and the result would change K0's rehearsal
choice. W0a is also conditional and must not delay an F0 treatment whose
reward is already frozen.

Stop after each decision gate. Do not pre-build later curriculum, reward, or
architecture machinery while an earlier result can invalidate it.

## 18. Definition of done for this backlog

The training redesign is ready for a generalist confirmation only when:

- termination, completion, reward, and evaluation share one legal task
  contract;
- reward and termination histories are globally reduced, stratified by
  outcome, and recomputable from machine-readable receipts;
- every active transition conserves world plus carried soil with no clipping,
  deletion, creation, or overflow;
- a correctly aimed dump stays inside the exact visible region and a wrong
  dump remains outside;
- relocation potential does not veto a physically valid action;
- full-reset and evaluator integrity contracts pass;
- one foundation and one trench identity are dynamically learnable;
- both quantitative easy families generalize separately from scratch;
- a scratch small multitask teacher and its grown medium student pass
  family/cell and retention gates;
- the map curriculum uses source-disjoint global promotion gates;
- procedural training diversity is sufficient and contains no silent repeated
  weighting;
- reward and partial-reset treatments remain separate;
- any architecture change is supported by a representation-specific failure;
  and
- the sealed bank remains untouched until model selection.
