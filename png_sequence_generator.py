#!/usr/bin/env python3
"""
Script to generate a file from a sequence of PNG files with naming pattern:
traversability_mask_partition_X_step_X.png
"""

import os
import re
import glob
from PIL import Image
import argparse
from typing import List, Tuple

def find_png_files(directory: str, pattern: str = "traversability_mask_partition_*_step_*.png") -> List[str]:
    """
    Find all PNG files matching the specified pattern and sort them numerically.
    
    Args:
        directory: Directory to search for PNG files
        pattern: File pattern to match
    
    Returns:
        List of sorted file paths
    """
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    
    # Sort files numerically by partition and step numbers
    def extract_numbers(filename):
        match = re.search(r'partition_(\d+)_step_(\d+)', filename)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return (0, 0)
    
    files.sort(key=extract_numbers)
    return files

def create_gif_animation(image_files: List[str], output_path: str, duration: int = 500) -> None:
    """
    Create a GIF animation from a sequence of PNG files.
    
    Args:
        image_files: List of image file paths
        output_path: Output GIF file path
        duration: Duration between frames in milliseconds
    """
    if not image_files:
        print("No image files found!")
        return
    
    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            # Convert to RGB if necessary (GIF doesn't support RGBA)
            if img.mode == 'RGBA':
                # Create a white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
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
        print(f"GIF animation saved to: {output_path}")
    else:
        print("No valid images found to create GIF!")

def create_grid_composite(image_files: List[str], output_path: str, cols: int = None) -> None:
    """
    Create a grid composite image from a sequence of PNG files.
    
    Args:
        image_files: List of image file paths
        output_path: Output image file path
        cols: Number of columns in the grid (auto-calculated if None)
    """
    if not image_files:
        print("No image files found!")
        return
    
    # Load all images
    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            images.append(img)
            print(f"Loaded: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if not images:
        print("No valid images found!")
        return
    
    # Calculate grid dimensions
    num_images = len(images)
    if cols is None:
        cols = int(num_images ** 0.5)
        if cols * cols < num_images:
            cols += 1
    
    rows = (num_images + cols - 1) // cols
    
    # Get image dimensions (assuming all images are the same size)
    img_width, img_height = images[0].size
    
    # Create composite image
    composite_width = cols * img_width
    composite_height = rows * img_height
    composite = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
    
    # Paste images into grid
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
    print(f"Grid composite saved to: {output_path} ({cols}x{rows} grid)")

def create_horizontal_strip(image_files: List[str], output_path: str) -> None:
    """
    Create a horizontal strip from a sequence of PNG files.
    
    Args:
        image_files: List of image file paths
        output_path: Output image file path
    """
    if not image_files:
        print("No image files found!")
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
        print("No valid images found!")
        return
    
    # Calculate total width and max height
    total_width = sum(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)
    
    # Create composite image
    composite = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    # Paste images horizontally
    x_offset = 0
    for img in images:
        if img.mode == 'RGBA':
            composite.paste(img, (x_offset, 0), img)
        else:
            composite.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    
    composite.save(output_path)
    print(f"Horizontal strip saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate a file from PNG sequence")
    parser.add_argument("directory", help="Directory containing PNG files")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    parser.add_argument("-t", "--type", choices=["gif", "grid", "strip"], 
                       default="gif", help="Output type (default: gif)")
    parser.add_argument("-d", "--duration", type=int, default=500,
                       help="GIF frame duration in milliseconds (default: 500)")
    parser.add_argument("-c", "--cols", type=int, help="Number of columns for grid layout")
    parser.add_argument("-p", "--pattern", default="traversability_mask_partition_*_step_*.png",
                       help="File pattern to match")
    
    args = parser.parse_args()
    
    # Find PNG files
    print(f"Searching for files in: {args.directory}")
    print(f"Pattern: {args.pattern}")
    
    png_files = find_png_files(args.directory, args.pattern)
    
    if not png_files:
        print("No PNG files found matching the pattern!")
        return
    
    print(f"Found {len(png_files)} files:")
    for file_path in png_files[:5]:  # Show first 5 files
        print(f"  - {os.path.basename(file_path)}")
    if len(png_files) > 5:
        print(f"  ... and {len(png_files) - 5} more")
    
    # Generate output based on type
    if args.type == "gif":
        if not args.output.lower().endswith('.gif'):
            args.output += '.gif'
        create_gif_animation(png_files, args.output, args.duration)
    elif args.type == "grid":
        if not any(args.output.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            args.output += '.png'
        create_grid_composite(png_files, args.output, args.cols)
    elif args.type == "strip":
        if not any(args.output.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            args.output += '.png'
        create_horizontal_strip(png_files, args.output)

if __name__ == "__main__":
    main()
