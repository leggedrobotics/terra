# Env Generation

This folder contains the essential tools for generating maps to train Terra agents. It leverages both procedurally generated environments and real-world building footprints.

## Available Map Types

- **Foundations**: Downloaded from OpenStreetMap and projected onto a grid map.
- **Trenches**: Procedurally generated trenches featuring 1, 2, or 3 axes, along with obstacles, no-dumping zones, and terminal dumping constraints.

## Generating Maps

### Step 1: Generate Foundation Maps

1. Download and generate the foundation maps:
    ```bash
    python generate_foundations.py
    ```
    If you prefer not to download the entire set of foundations, you can interrupt the script and rerun it. The script will automatically format the maps correctly.

### Step 2: Create Training Data

1. Generate procedural trenches, add constraints and obstacles, and reformat the maps for Terra use:
    ```bash
    python create_train_data.py
    ```

### Step 3: Verify Map Generation

1. Ensure the maps are correctly generated by running:
    ```bash
    DATASET_PATH="<path_to_terra>/terra/digbench/data/train/" DATASET_SIZE=<N> python -m terra.viz.play
    ```

Replace `<path_to_terra>` with the actual path to your Terra installation and `<N>` with the desired dataset size.
sql
