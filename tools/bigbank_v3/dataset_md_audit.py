#!/usr/bin/env python3
"""docs/DATASET.md identity/split audit (origin/main), with only the directory list changed.

train = the new pooled bank; evaluation = the V8 R2 main and capability panels
(as in DATASET.md) plus the gate_main panels and the TTC known-geometry panels.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import numpy as np

bank = Path('/home/lorenzo/moleworks/.artifacts/'
            'terra_v8_r2_training_inputs_20260810/treatment_bank')
enriched = Path('/home/lorenzo/moleworks/.artifacts/terra_v8_trench_finite_enriched_20260819')
ttc = Path('/home/lorenzo/moleworks/.artifacts/terra_test_time_compute_20260921/adaptation/maps')
new_bank = Path('/home/lorenzo/moleworks/.artifacts/terra_gru_bigbank_20260923/bank/train_v3_generalist_512')
directories = [('train', new_bank)]
directories += [(split, bank / 'evaluation' / family / split)
                for family in ('main', 'capability_floor')
                for split in ('promotion', 'development', 'sealed')]
directories += [(f'gate_main_{split}', enriched / 'evaluation' / 'gate_main' / split)
                for split in ('promotion', 'development', 'sealed')]
directories += [('ttc_development', ttc / case / 'eval')
                for case in ('trn-straight-side1', 'trn-tee-side2', 'trn-net4-side1-road')]
counts = Counter()
groups = defaultdict(lambda: defaultdict(set))
scenario_slots = defaultdict(list)
for split, directory in directories:
    for line in (directory / 'manifest.jsonl').read_text().splitlines():
        row = json.loads(line)
        slot = row['slot_index']
        dig = np.load(directory / 'images' / f'img_{slot}.npy') < 0
        assert dig.shape == (64, 64)
        counts[split] += 1
        groups['source'][split].add(row['source_id'])
        groups['scenario'][split].add(row['scenario_id'])
        groups['dig'][split].add(sha256(dig.tobytes()).hexdigest())
        scenario_slots[split, row['scenario_id']].append((directory, slot))
for split, count in counts.items():
    print(split, 'slots', count,
          {kind: len(by_split[split]) for kind, by_split in groups.items()})
for kind, by_split in groups.items():
    for first, second in combinations(by_split, 2):
        print(kind, first, second, 'overlap',
              len(by_split[first] & by_split[second]))
for (split, scenario), slots in scenario_slots.items():
    if len(slots) > 1:
        for (first, a), (second, b) in combinations(slots, 2):
            equal = all(np.array_equal(
                np.load(first / layer / f'img_{a}.npy'),
                np.load(second / layer / f'img_{b}.npy'))
                for layer in ('images', 'occupancy', 'dumpability',
                              'actions', 'distance'))
            print('duplicate', split, first.name, a, second.name, b,
                  'all_five_arrays_equal', equal)
