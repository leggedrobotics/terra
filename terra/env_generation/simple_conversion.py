#!/usr/bin/env python3
"""
Simple script to convert foundations_dumpzones to Terra format using default parameters.
"""

import os
import sys
from pathlib import Path

def simple_conversion():
    """Convert foundations_dumpzones using default parameters."""
    print("🔄 Simple conversion of foundations_dumpzones...")
    
    # Define paths
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(package_dir, "data", "terra")
    
    print(f"Data directory: {data_dir}")
    
    # Check if foundations_dumpzones exists
    foundations_dumpzones_dir = os.path.join(data_dir, "foundations_dumpzones")
    if not os.path.exists(foundations_dumpzones_dir):
        print(f"❌ foundations_dumpzones directory not found: {foundations_dumpzones_dir}")
        return False
    
    print(f"✅ Found foundations_dumpzones directory")
    
    try:
        # Import the conversion function
        from convert_to_terra import generate_foundations_dumpzones_terra
        
        # Use default parameters
        size = (50, 50)  # Default size
        n_imgs = 1000    # Default number of images
        
        print(f"Converting with defaults: size={size}, n_imgs={n_imgs}")
        
        # Run the conversion
        generate_foundations_dumpzones_terra(data_dir, size, n_imgs)
        
        print("✅ Conversion completed!")
        
        # Check the results
        train_dir = os.path.join(data_dir, "train", "foundations_dumpzones")
        if os.path.exists(train_dir):
            files = os.listdir(train_dir)
            print(f"Train directory contains: {files}")
            
            # Check each subdirectory
            for subdir in files:
                subdir_path = os.path.join(train_dir, subdir)
                if os.path.isdir(subdir_path):
                    subdir_files = os.listdir(subdir_path)
                    print(f"  {subdir}: {len(subdir_files)} files")
                    if subdir_files:
                        print(f"    Sample: {subdir_files[:3]}")
        else:
            print("❌ Train directory not found after conversion")
            return False
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_generate_dataset():
    """Test using the generate_dataset_terra_format function directly."""
    print("\n🔄 Testing with generate_dataset_terra_format...")
    
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(package_dir, "data", "terra")
    
    try:
        from convert_to_terra import generate_dataset_terra_format
        
        # Use default parameters
        size = (50, 50)
        n_imgs = 1000
        map_types = ["foundations_dumpzones"]  # Only convert foundations_dumpzones
        
        print(f"Running generate_dataset_terra_format with: size={size}, n_imgs={n_imgs}, map_types={map_types}")
        
        generate_dataset_terra_format(data_dir, size, n_imgs, map_types)
        
        print("✅ generate_dataset_terra_format completed!")
        
        # Check results
        train_dir = os.path.join(data_dir, "train", "foundations_dumpzones")
        if os.path.exists(train_dir):
            files = os.listdir(train_dir)
            print(f"Train directory contains: {files}")
        else:
            print("❌ Train directory not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ generate_dataset_terra_format failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the simple conversion."""
    print("🚀 Simple foundations_dumpzones conversion test")
    
    # Test the dedicated function
    success1 = simple_conversion()
    
    # Test the general function
    success2 = test_with_generate_dataset()
    
    if success1 and success2:
        print("\n🎉 All conversions successful!")
    else:
        print("\n❌ Some conversions failed!")

if __name__ == "__main__":
    main() 