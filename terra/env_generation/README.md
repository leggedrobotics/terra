# Env Generation

This folder contains the essential tools for generating maps to train Terra agents. It leverages both procedurally generated environments and real-world building footprints.

## Available Map Types

- **Foundations**: Downloaded from OpenStreetMap and projected onto a grid map.
- **Trenches**: Procedurally generated trenches featuring 1, 2, or 3 axes, along with obstacles, no-dumping zones, and terminal dumping constraints.

## Generating Maps

### Step 1: Create Training Maps

1. Generate procedural trenches, add constraints and obstacles, and reformat the maps for Terra use:
    ```bash
    python generate_dataset.py
    ```
This will create a data/train folder which contains the maps used during training.

### Step 3: Verify Map Generation

1. Ensure the maps are correctly generated by running:
    ```bash
    DATASET_PATH="<path_to_terra>/terra/digbench/data/train/" DATASET_SIZE=<N> python -m terra.viz.play
    ```
For example:
```
DATASET_PATH=<parent_dir>/terra/data/terra/train DATASET_SIZE=24 python -m terra.viz.play
```
Replace `<path_to_terra>` with the actual path to your Terra installation and `<N>` with the desired dataset size.

## Data Generation Workflows In Detail

This section provides a deeper understanding of how each type of training data is generated and processed in the Terra system.

### Foundations Data Workflow

The foundations data is based on real-world building footprints from OpenStreetMap:

1. **Download and Processing**:
   - `generate_foundations.py` downloads building footprints from OpenStreetMap using the specified bounding box
   - Buildings are projected onto a grid with configurable resolution
   - Images undergo preprocessing including padding, hole filling, and filtering

2. **convert_to_terra**:
   - The `create_foundations` function in `create_train_data.py` handles:
     - Downsampling of images to fit maximum size requirements
     - Converting images to the Terra format
     - Generating occupancy and dumpability maps
   - The convert_to_terra module applies final transformations to make the data usable for training

3. **Configuration**:
   - Parameters are loaded from the config YAML file, including:
     - Resolution and size constraints
     - Dataset paths
     - Optional obstacle and non-dumpable zone parameters

### Trenches Data Workflow

Trenches are procedurally generated environments for excavation tasks:

1. **Generation**:
   - `create_procedural_trenches` in `create_train_data.py` handles generation
   - The `generate_trenches_v2` function in `procedural_data.py` creates trenches with 1, 2, or 3 axes
   - Trenches are organized in different difficulty levels based on configuration

2. **Feature Addition**:
   - Obstacles are added with configurable parameters (number, size)
   - No-dumping zones are placed with constraints
   - Terminal dumping constraints are applied

3. **Data Organization**:
   - Trenches are saved in folders organized by difficulty level
   - Each trench includes image data, metadata, occupancy, and dumpability maps
   - The `generate_trenches_terra` function in `convert_to_terra.py` converts all data to the Terra format

### Curriculum Generation

For structured training progression:

1. **Generating Curriculum Data**:
   - `generate_curriculum.py` creates a progression of environments with increasing difficulty
   - Different environment types can be integrated into the curriculum
   - Each stage is stored in appropriately named folders

2. **Usage**:
   - Configure the curriculum in the config file
   - Run the curriculum generator
   - The resulting data follows a progression suitable for staged training

### Data Format Conversion

All generated data undergoes format conversion for training:

1. **Conversion Process**:
   - `convert_to_terra.py` contains functions to convert all data to the Terra format
   - `generate_dataset_terra_format` converts data to multiple resolutions
   - Images, occupancy maps, and dumpability maps are all properly formatted

2. **Output Structure**:
   - Final data is organized in the `/data/terra/train/` directory
   - Subdirectories include foundations, trenches (with difficulty levels) and custom maps
   - Each environment has consistent format and metadata
