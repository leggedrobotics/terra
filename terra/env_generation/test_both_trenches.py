#!/usr/bin/env python
"""
Test script to verify that both regular trenches and trenches with dump zones can be generated.
"""

import os
import sys
import yaml
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from terra.env_generation.create_train_data import create_procedural_trenches, create_procedural_trenches_with_dumpzones

def test_both_trench_types():
    """Test both regular trenches and trenches with dump zones generation."""
    
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
    
    print("Testing both trench generation types...")
    print(f"Config: {test_config}")
    
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Test 1: Regular trenches
    print("\n=== Testing Regular Trenches ===")
    try:
        create_procedural_trenches(test_config)
        
        # Check if the output directories were created
        output_dir = os.path.join(package_dir, "data", "terra", "trenches", "single")
        
        if os.path.exists(output_dir):
            print(f"✓ Successfully created regular trenches: {output_dir}")
            
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
            print(f"✗ Regular trenches directory not created: {output_dir}")
            return False
            
    except Exception as e:
        print(f"✗ Error during regular trench generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Trenches with dump zones
    print("\n=== Testing Trenches with Dump Zones ===")
    try:
        create_procedural_trenches_with_dumpzones(test_config)
        
        # Check if the output directories were created
        output_dir = os.path.join(package_dir, "data", "terra", "trenches", "single_dumpzone")
        
        if os.path.exists(output_dir):
            print(f"✓ Successfully created trenches with dump zones: {output_dir}")
            
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
            print(f"✗ Trenches with dump zones directory not created: {output_dir}")
            return False
            
    except Exception as e:
        print(f"✗ Error during trenches with dump zones generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n=== Summary ===")
    print("✓ Both trench types generated successfully!")
    print("✓ Regular trenches: data/terra/trenches/single/")
    print("✓ Trenches with dump zones: data/terra/trenches/single_dumpzone/")
    
    return True

if __name__ == "__main__":
    success = test_both_trench_types()
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Tests failed!")
        sys.exit(1) 