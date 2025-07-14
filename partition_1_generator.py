#!/usr/bin/env python3
"""
Script to generate a file from PNG sequence files for PARTITION 1 only.
Processes files with pattern: traversability_mask_partition_1_step_X.png
"""

import os
import re
import glob
from PIL import Image
import argparse
from typing import List

def find_partition_1_files(directory: str) -> List[str]:
    """
    Find all PNG files for partition 1 and sort them by step number.
    
    Args:
        directory: Directory to search for PNG files
    
    Returns:
        List of sorted file paths for partition 1
    """
    pattern = os.path.join(directory, "traversability_partition_1_step_*.png")
    files = glob.glob(pattern)
    
    # Sort files numerically by step number
    def extract_step_number(filename):
        match = re.search(r'partition_1_step_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    files.sort(key=extract_step_number)
    return files

def create_gif_animation(image_files: List[str], output_path: str, duration: int = 500) -> None:
    """Create a GIF animation from partition 1 PNG files."""
    if not image_files:
        print("No partition 1 image files found!")
        return
    
    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            print(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"Partition 1 GIF animation saved to: {output_path}")
        print(f"Total frames: {len(images)}")
    else:
        print("No valid partition 1 images found to create GIF!")

def create_grid_composite(image_files: List[str], output_path: str, cols: int = None) -> None:
    """Create a grid composite from partition 1 PNG files."""
    if not image_files:
        print("No partition 1 image files found!")
        return
    
    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            images.append(img)
            print(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not images:
        print("No valid partition 1 images found!")
        return
    
    num_images = len(images)
    if cols is None:
        cols = int(num_images ** 0.5)
        if cols * cols < num_images:
            cols += 1
    
    rows = (num_images + cols - 1) // cols
    img_width, img_height = images[0].size
    
    composite_width = cols * img_width
    composite_height = rows * img_height
    composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * img_width
        y = row * img_height
        
        if img.mode == 'RGBA':
            composite.paste(img, (x, y), img)
        else:
            composite.paste(img, (x, y))
    
    composite.save(output_path)
    print(f"Partition 1 grid composite saved to: {output_path} ({cols}x{rows} grid)")
    print(f"Total images: {len(images)}")

def create_horizontal_strip(image_files: List[str], output_path: str) -> None:
    """Create a horizontal strip from partition 1 PNG files."""
    if not image_files:
        print("No partition 1 image files found!")
        return
    
    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            images.append(img)
            print(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not images:
        print("No valid partition 1 images found!")
        return
    
    total_width = sum(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)
    
    composite = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    x_offset = 0
    for img in images:
        if img.mode == 'RGBA':
            composite.paste(img, (x_offset, 0), img)
        else:
            composite.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    composite.save(output_path)
    print(f"Partition 1 horizontal strip saved to: {output_path}")
    print(f"Total images: {len(images)}")

def main():
    parser = argparse.ArgumentParser(description="Generate file from PARTITION 1 PNG sequence")
    parser.add_argument("directory", help="Directory containing PNG files")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("-t", "--type", choices=["gif", "grid", "strip"], 
                       default="gif", help="Output type (default: gif)")
    parser.add_argument("-d", "--duration", type=int, default=500,
                       help="GIF frame duration in milliseconds (default: 500)")
    parser.add_argument("-c", "--cols", type=int, help="Number of columns for grid layout")
    
    args = parser.parse_args()
    
    print(f"Processing PARTITION 1 files from: {args.directory}")
    print("Looking for pattern: traversability_partition_1_step_*.png")
    
    png_files = find_partition_1_files(args.directory)
    
    if not png_files:
        print("No partition 1 PNG files found!")
        return
    
    print(f"Found {len(png_files)} partition 1 files:")
    for file_path in png_files[:10]:  # Show first 10 files
        step_match = re.search(r'step_(\d+)', os.path.basename(file_path))
        step_num = step_match.group(1) if step_match else "?"
        print(f"  - Step {step_num}: {os.path.basename(file_path)}")
    if len(png_files) > 10:
        print(f"  ... and {len(png_files) - 10} more")
    
    # Generate output based on type
    if args.type == "gif":
        if not args.output.lower().endswith('.gif'):
            args.output = args.output.replace('.', '_partition_1.') if '.' in args.output else args.output + '_partition_1.gif'
        create_gif_animation(png_files, args.output, args.duration)
    elif args.type == "grid":
        if not any(args.output.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            args.output += '_partition_1.png'
        create_grid_composite(png_files, args.output, args.cols)
    elif args.type == "strip":
        if not any(args.output.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            args.output += '_partition_1.png'
        create_horizontal_strip(png_files, args.output)

if __name__ == "__main__":
    main()
