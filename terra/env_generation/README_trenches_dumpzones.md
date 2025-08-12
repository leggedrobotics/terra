# Trenches with Dump Zones

This document explains how to generate trenches with specific dump zones, similar to the foundations with dump zones functionality.

## Overview

The trenches with dump zones feature allows you to generate trench environments that include a single, strategically placed dump zone. This is useful for multi-agent training scenarios where you want to control where the skidsteer can dump material.

## Key Features

- **Non-destructive**: Original trenches without dump zones are still generated normally
- **Dual functionality**: Both `create_procedural_trenches()` and `create_procedural_trenches_with_dumpzones()` are available
- **Single dump zone**: Each trench gets exactly one dump zone placed at the border
- **Smart placement**: Dump zones avoid overlapping with dig zones
- **Fallback logic**: If placement fails, the system tries with smaller dump zones
- **Separate folders**: Output goes to `trenches/{level}_dumpzone/` folders

## Usage

### Command Line

To generate trenches with dump zones, use the `--trenches-dumpzones` flag:

```bash
# Generate only trenches with dump zones
python generate_dataset.py --trenches-dumpzones

# Generate both regular trenches and trenches with dump zones
python generate_dataset.py --trenches --trenches-dumpzones

# Generate all map types including trenches with dump zones
python generate_dataset.py --all --trenches-dumpzones
```

### Programmatic Usage

```python
from terra.env_generation.create_train_data import create_procedural_trenches_with_dumpzones

# Load your config
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Generate trenches with dump zones
create_procedural_trenches_with_dumpzones(config)
```

## Output Structure

The trenches with dump zones are saved in the following structure:

```
data/terra/trenches/
├── single/                  # Original trenches (unchanged)
├── double/                  # Original trenches (unchanged)
├── triple/                  # Original trenches (unchanged)
├── single_dumpzone/         # New: trenches with dump zones
├── double_dumpzone/         # New: trenches with dump zones
└── triple_dumpzone/         # New: trenches with dump zones
```

Each `*_single_dumpzone` folder contains:
- `images/` - The generated trench images with dump zones
- `metadata/` - Metadata for each image
- `occupancy/` - Occupancy maps
- `dumpability/` - Dumpability maps

## Dump Zone Parameters

The dump zones are generated with the following default parameters:

- **Number**: Exactly 1 dump zone per trench
- **Size**: 10-13 tiles (configurable)
- **Placement**: At the border with 3-tile offset
- **Avoidance**: No overlap with dig zones

## Terra Format Conversion

When you run with `--terra-format` (default), the trenches with dump zones are automatically converted to Terra format and saved in:

```
data/terra/train/trenches/
├── single_dumpzone/
├── double_dumpzone/
└── triple_dumpzone/
```

## Configuration

The trenches with dump zones use the same configuration as regular trenches, with additional dump zone parameters:

```yaml
trenches:
  difficulty_levels: ["single", "double", "triple"]
  trenches_per_level: [(1, 2), (2, 3), (3, 4)]
  img_edge_min: 32
  img_edge_max: 48
  trench_dims:
    single:
      min_ratio: [0.1, 0.1]
      max_ratio: [0.3, 0.3]
      diagonal: false
    # ... other levels
  n_obs_min: 0
  n_obs_max: 1
  size_obstacle_min: 2
  size_obstacle_max: 4
  n_nodump_min: 0
  n_nodump_max: 1
  size_nodump_min: 2
  size_nodump_max: 4
```

## Testing

You can test the functionality using the provided test scripts:

```bash
cd terra/terra/env_generation

# Test only trenches with dump zones
python test_trenches_dumpzones.py

# Test both regular trenches and trenches with dump zones
python test_both_trenches.py
```

These will generate test trenches and verify the output structure.

## Implementation Details

### Key Functions

1. **`create_procedural_trenches_with_dumpzones(config)`**: Main entry point
2. **`generate_trenches_with_dumpzones(...)`**: Core generation logic
3. **`create_single_dump_zone_trenches(...)`**: Dump zone placement logic

### Differences from Regular Trenches

1. **Background**: Uses neutral background instead of dumping background
2. **Dump zone placement**: Adds exactly one dump zone per trench
3. **Folder naming**: Appends `_single_dumpzone` to difficulty levels
4. **Terra conversion**: Uses dedicated conversion function

### Fallback Logic

If dump zone placement fails:
1. Try with original size (100 attempts)
2. Try with size -3 (100 attempts)
3. Try with size -5 (100 attempts)
4. Raise error if all attempts fail

## Compatibility

- **Backward compatible**: Existing trench generation is unchanged
- **Config compatible**: Uses same configuration structure
- **Terra format compatible**: Converts to standard Terra format
- **Multi-agent compatible**: Designed for excavator-skidsteer scenarios 