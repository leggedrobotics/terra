#!/usr/bin/env python
"""
Test script for trenches with dump zones functionality.
This script tests the new trenches with dump zones generation without affecting existing trenches.
"""

import os
import sys
import yaml
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from terra.env_generation.create_train_data import create_procedural_trenches_with_dumpzones

def test_trenches_with_dumpzones():
    """Test the trenches with dump zones generation."""
    
    # Create a minimal test config
    test_config = {
        "resolution": 1,
        "n_imgs": 2,  # Just generate 2 images for testing
        "trenches": {
            "difficulty_levels": ["single"],
            "trenches_per_level": [(1, 2)],  # 1-2 trenches for single level
            "img_edge_min": 32,
            "img_edge_max": 48,
            "trench_dims": {
                "single": {
                    "min_ratio": [0.1, 0.1],
                    "max_ratio": [0.3, 0.3],
                    "diagonal": False
                }
            },
            "n_obs_min": 0,
            "n_obs_max": 1,
            "size_obstacle_min": 2,
            "size_obstacle_max": 4,
            "n_nodump_min": 0,
            "n_nodump_max": 1,
            "size_nodump_min": 2,
            "size_nodump_max": 4
        }
    }
    
    print("Testing trenches with dump zones generation...")
    print(f"Config: {test_config}")
    
    try:
        # Generate trenches with dump zones
        create_procedural_trenches_with_dumpzones(test_config)
        
        # Check if the output directories were created
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(package_dir, "data", "terra", "trenches", "single_dumpzone")
        
        if os.path.exists(output_dir):
            print(f"✓ Successfully created output directory: {output_dir}")
            
            # Check for subdirectories
            subdirs = ["images", "metadata", "occupancy", "dumpability"]
            for subdir in subdirs:
                subdir_path = os.path.join(output_dir, subdir)
                if os.path.exists(subdir_path):
                    files = os.listdir(subdir_path)
                    print(f"✓ {subdir} directory created with {len(files)} files")
                else:
                    print(f"✗ {subdir} directory not found")
        else:
            print(f"✗ Output directory not created: {output_dir}")
            
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("Test completed!")
    return True

if __name__ == "__main__":
    success = test_trenches_with_dumpzones()
    if success:
        print("✓ Test passed!")
        sys.exit(0)
    else:
        print("✗ Test failed!")
        sys.exit(1) 