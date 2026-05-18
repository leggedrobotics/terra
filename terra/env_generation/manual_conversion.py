#!/usr/bin/env python3
"""
Manual conversion script for foundations_dumpzones.
"""

import os
import sys
from pathlib import Path

def main():
    """Manually convert foundations_dumpzones to Terra format."""
    print("🔄 Manual conversion of foundations_dumpzones...")
    
    # Define paths
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(package_dir, "data", "terra")
    
    print(f"Data directory: {data_dir}")
    
    # Check if foundations_dumpzones exists
    foundations_dumpzones_dir = os.path.join(data_dir, "foundations_dumpzones")
    if not os.path.exists(foundations_dumpzones_dir):
        print(f"❌ foundations_dumpzones directory not found: {foundations_dumpzones_dir}")
        return False
    
    print(f"✅ foundations_dumpzones directory found: {foundations_dumpzones_dir}")
    
    # Check if train directory exists
    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(train_dir):
        print(f"❌ Train directory not found: {train_dir}")
        return False
    
    print(f"✅ Train directory found: {train_dir}")
    
    # Create foundations_dumpzones train directory if it doesn't exist
    foundations_dumpzones_train = os.path.join(train_dir, "foundations_dumpzones")
    if not os.path.exists(foundations_dumpzones_train):
        print(f"🔄 Creating foundations_dumpzones train directory...")
        os.makedirs(foundations_dumpzones_train, exist_ok=True)
        print(f"✅ Created: {foundations_dumpzones_train}")
    
    # Create subdirectories
    subdirs = ["images", "metadata", "occupancy", "dumpability"]
    for subdir in subdirs:
        subdir_path = os.path.join(foundations_dumpzones_train, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
            print(f"✅ Created subdirectory: {subdir_path}")
    
    # Now try to run the actual conversion
    try:
        # Import the conversion function
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from convert_to_terra import _convert_all_imgs_to_terra
        
        print("🔄 Running conversion...")
        
        # Set up paths
        img_folder = Path(foundations_dumpzones_dir) / "images"
        metadata_folder = Path(foundations_dumpzones_dir) / "metadata"
        occupancy_folder = Path(foundations_dumpzones_dir) / "occupancy"
        dumpability_folder = Path(foundations_dumpzones_dir) / "dumpability"
        destination_folder = Path(foundations_dumpzones_train)
        
        print(f"Source folders:")
        print(f"  Images: {img_folder}")
        print(f"  Metadata: {metadata_folder}")
        print(f"  Occupancy: {occupancy_folder}")
        print(f"  Dumpability: {dumpability_folder}")
        print(f"Destination: {destination_folder}")
        
        # Run conversion
        _convert_all_imgs_to_terra(
            img_folder,
            metadata_folder,
            occupancy_folder,
            dumpability_folder,
            destination_folder,
            size=(50, 50),
            n_imgs=10,  # Convert first 10 images
            all_dumpable=False,  # Foundations with dump zones use specific dump zones
            copy_metadata=True,
            downsample=False,
            has_dumpability=True,
            center_padding=False,
            actions_folder=None,
        )
        
        print("✅ Conversion completed successfully!")
        
        # Check if files were created
        for subdir in subdirs:
            subdir_path = os.path.join(foundations_dumpzones_train, subdir)
            files = os.listdir(subdir_path)
            print(f"  {subdir}: {len(files)} files")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This might be due to missing dependencies (OpenCV, NumPy, etc.)")
        return False
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 