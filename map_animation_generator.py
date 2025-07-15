#!/usr/bin/env python3
"""
Enhanced script to generate side-by-side animations comparing partition 0 and partition 1.
Creates a master grid showing all map types in two columns: Partition 0 | Partition 1

Usage examples:
python map_animation_generator.py /path/to/images -o master_comparison.gif
python map_animation_generator.py /path/to/images -o comparison.gif -d 300 --maps target action traversability
python map_animation_generator.py /path/to/images -o output.gif --single-map traversability
"""

import os
import re
import glob
import argparse
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Dict, Tuple
import sys
from collections import defaultdict

class MasterMapAnimationGenerator:
    """Class to handle master animation generation comparing partition 0 and partition 1."""
    
    COMMON_MAP_TYPES = [
        'target', 'action', 'dumpability', 'dumpability_init', 
        'padding', 'traversability'
    ]
    
    def __init__(self, directory: str):
        self.directory = directory
        self.validate_directory()
    
    def validate_directory(self):
        """Validate input directory."""
        if not os.path.exists(self.directory):
            raise ValueError(f"Directory '{self.directory}' does not exist")
    
    def discover_map_types(self) -> List[str]:
        """Discover all available map types in the directory."""
        pattern = os.path.join(self.directory, "*.png")
        all_files = glob.glob(pattern)
        
        map_types = set()
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            # Try different patterns to extract map type
            patterns = [
                r'(\w+)(?:_mask)?_partition_\d+',
                r'(\w+)_p\d+_step',
                r'(\w+)_\d+_step'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    map_type = match.group(1)
                    if map_type not in ['step', 'partition']:  # Filter out common false positives
                        map_types.add(map_type)
                    break
        
        return sorted(list(map_types))
    
    def find_files_for_map_and_partition(self, map_type: str, partition: int) -> List[str]:
        """Find all PNG files for a specific map type and partition."""
        patterns = [
            f"{map_type}_partition_{partition}_step_*.png",
            f"{map_type}_mask_partition_{partition}_step_*.png",
            f"{map_type}_{partition}_step_*.png",
            f"{map_type}_p{partition}_step_*.png",
            f"{map_type}_partition{partition}_step_*.png"
        ]
        
        files = []
        for pattern_template in patterns:
            pattern = os.path.join(self.directory, pattern_template)
            found_files = glob.glob(pattern)
            if found_files:
                files = found_files
                break
        
        if not files:
            # Broader search
            pattern = os.path.join(self.directory, f"*{map_type}*partition*{partition}*.png")
            files = glob.glob(pattern)
        
        # Sort files numerically by step number
        def extract_step_number(filename):
            patterns = [r'step_(\d+)', r'step(\d+)', r'_(\d+)\.png']
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    return int(match.group(1))
            return 0
        
        files.sort(key=extract_step_number)
        return files
    
    def load_images_from_files(self, file_paths: List[str]) -> List[Image.Image]:
        """Load images from file paths and convert to RGB."""
        images = []
        for file_path in file_paths:
            try:
                img = Image.open(file_path)
                
                # Convert to RGB if necessary
                if img.mode == 'RGBA':
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])
                    img = background
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                images.append(img)
            except Exception as e:
                print(f"Warning: Error loading {file_path}: {e}")
        
        return images
    
    def create_labeled_image(self, img: Image.Image, label: str, 
                           font_size: int = None) -> Image.Image:
        """Add a label to the top of an image with improved readability."""
        # Auto-calculate font size based on image width
        if font_size is None:
            font_size = max(12, min(24, img.width // 20))
        
        # Calculate label area - make it taller for better readability
        label_height = int(font_size * 2.5)
        new_height = img.height + label_height
        
        # Create new image with label space (darker background for contrast)
        labeled_img = Image.new('RGB', (img.width, new_height), (50, 50, 50))
        
        # Paste original image below label area
        labeled_img.paste(img, (0, label_height))
        
        # Add label text with better formatting
        draw = ImageDraw.Draw(labeled_img)
        
        # Try to get a better font
        font = None
        try:
            # Try to load a TrueType font if available
            if os.name == 'nt':  # Windows
                font_paths = [
                    'C:/Windows/Fonts/arial.ttf',
                    'C:/Windows/Fonts/calibri.ttf',
                    'C:/Windows/Fonts/tahoma.ttf'
                ]
            else:  # Unix/Linux/Mac
                font_paths = [
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                    '/System/Library/Fonts/Arial.ttf',
                    '/usr/share/fonts/TTF/arial.ttf'
                ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    break
        except:
            pass
        
        # Fallback to default font with larger size
        if font is None:
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        # Shorten label if too long
        max_chars = max(10, img.width // (font_size // 2))
        if len(label) > max_chars:
            # Split on " - " and take parts
            parts = label.split(' - ')
            if len(parts) == 2:
                map_type, partition = parts
                # Abbreviate map type if needed
                if len(map_type) > max_chars - 6:
                    map_type = map_type[:max_chars-6] + '.'
                label = f"{map_type}\n{partition}"
            else:
                label = label[:max_chars-3] + '...'
        
        # Handle multi-line labels
        lines = label.split('\n')
        
        # Calculate total text height
        line_height = font_size + 4
        total_text_height = len(lines) * line_height
        
        # Position text in center of label area
        start_y = (label_height - total_text_height) // 2
        
        for i, line in enumerate(lines):
            # Calculate text position (centered horizontally)
            if font:
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
            else:
                text_width = len(line) * (font_size // 2)
            
            text_x = (img.width - text_width) // 2
            text_y = start_y + (i * line_height)
            
            # Add white outline for better visibility
            outline_color = (255, 255, 255)
            text_color = (255, 255, 255)
            
            # Draw text outline
            for adj_x, adj_y in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                draw.text((text_x + adj_x, text_y + adj_y), line, 
                         fill=(0, 0, 0), font=font)
            
            # Draw main text
            draw.text((text_x, text_y), line, fill=text_color, font=font)
        
        return labeled_img
    
    def resize_images_to_match(self, images: List[Image.Image]) -> List[Image.Image]:
        """Resize all images to match the smallest dimensions."""
        if not images:
            return []
        
        # Find minimum dimensions
        min_width = min(img.width for img in images)
        min_height = min(img.height for img in images)
        
        # Resize all images
        resized_images = []
        for img in images:
            if img.size != (min_width, min_height):
                resized_img = img.resize((min_width, min_height), Image.Resampling.LANCZOS)
                resized_images.append(resized_img)
            else:
                resized_images.append(img)
        
        return resized_images
    
    def pad_frame_sequences(self, seq1: List[Image.Image], 
                           seq2: List[Image.Image]) -> Tuple[List[Image.Image], List[Image.Image]]:
        """Pad shorter sequence to match longer one by repeating last frame."""
        if len(seq1) == len(seq2):
            return seq1, seq2
        
        max_len = max(len(seq1), len(seq2))
        
        # Pad seq1 if shorter
        if len(seq1) < max_len:
            if seq1:  # If not empty
                last_frame = seq1[-1]
                seq1.extend([last_frame] * (max_len - len(seq1)))
            else:
                # Create blank frames if empty
                blank = Image.new('RGB', (100, 100), (200, 200, 200))
                seq1 = [blank] * max_len
        
        # Pad seq2 if shorter  
        if len(seq2) < max_len:
            if seq2:  # If not empty
                last_frame = seq2[-1]
                seq2.extend([last_frame] * (max_len - len(seq2)))
            else:
                # Create blank frames if empty
                blank = Image.new('RGB', (100, 100), (200, 200, 200))
                seq2 = [blank] * max_len
        
        return seq1, seq2
    
    def create_master_comparison_animation(self, map_types: List[str], output_path: str,
                                         duration: int = 500, max_width: int = 800,
                                         font_size: int = None, no_labels: bool = False) -> None:
        """Create master animation comparing all map types between partition 0 and 1."""
        print(f"Creating master comparison for map types: {', '.join(map_types)}")
        print("-" * 80)
        
        # Collect all image sequences
        all_sequences = []
        valid_map_types = []
        
        for map_type in map_types:
            print(f"\nProcessing {map_type}...")
            
            # Get files for both partitions
            files_p0 = self.find_files_for_map_and_partition(map_type, 0)
            files_p1 = self.find_files_for_map_and_partition(map_type, 1)
            
            print(f"  Partition 0: {len(files_p0)} files")
            print(f"  Partition 1: {len(files_p1)} files")
            
            if not files_p0 and not files_p1:
                print(f"  ⚠️  Skipping {map_type} - no files found for either partition")
                continue
            
            # Load images
            images_p0 = self.load_images_from_files(files_p0)
            images_p1 = self.load_images_from_files(files_p1)
            
            # Pad sequences to same length
            images_p0, images_p1 = self.pad_frame_sequences(images_p0, images_p1)
            
            # Resize to match
            all_images = images_p0 + images_p1
            all_images = self.resize_images_to_match(all_images)
            
            # Split back
            mid_point = len(all_images) // 2
            images_p0 = all_images[:mid_point]
            images_p1 = all_images[mid_point:]
            
            # Add labels with shorter, cleaner text
            if no_labels:
                labeled_p0 = images_p0
                labeled_p1 = images_p1
            else:
                labeled_p0 = [self.create_labeled_image(img, f"{map_type}\nP0", font_size) 
                             for img in images_p0]
                labeled_p1 = [self.create_labeled_image(img, f"{map_type}\nP1", font_size) 
                             for img in images_p1]
            
            all_sequences.append((labeled_p0, labeled_p1))
            valid_map_types.append(map_type)
            print(f"  ✓ Added {map_type} with {len(labeled_p0)} frames")
        
        if not all_sequences:
            print("❌ No valid map types found!")
            return
        
        print(f"\n🎬 Creating master grid with {len(valid_map_types)} map types...")
        
        # Determine grid layout
        num_rows = len(all_sequences)
        num_cols = 2  # Always 2 columns (partition 0 and 1)
        
        # Get frame count (should be same for all sequences)
        max_frames = max(len(seq[0]) for seq in all_sequences)
        
        # Get individual image dimensions
        sample_img = all_sequences[0][0][0]
        img_width, img_height = sample_img.size
        
        # Scale down if needed
        if img_width > max_width // 2:
            scale_factor = (max_width // 2) / img_width
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            
            # Rescale all images
            for i, (seq_p0, seq_p1) in enumerate(all_sequences):
                scaled_p0 = [img.resize((new_width, new_height), Image.Resampling.LANCZOS) 
                           for img in seq_p0]
                scaled_p1 = [img.resize((new_width, new_height), Image.Resampling.LANCZOS) 
                           for img in seq_p1]
                all_sequences[i] = (scaled_p0, scaled_p1)
            
            img_width, img_height = new_width, new_height
        
        # Create master frames
        master_width = num_cols * img_width
        master_height = num_rows * img_height
        
        master_frames = []
        
        for frame_idx in range(max_frames):
            master_frame = Image.new('RGB', (master_width, master_height), (255, 255, 255))
            
            for row_idx, (seq_p0, seq_p1) in enumerate(all_sequences):
                # Get current frame (or last frame if sequence is shorter)
                frame_p0 = seq_p0[min(frame_idx, len(seq_p0) - 1)]
                frame_p1 = seq_p1[min(frame_idx, len(seq_p1) - 1)]
                
                # Calculate positions
                y_pos = row_idx * img_height
                x_pos_p0 = 0
                x_pos_p1 = img_width
                
                # Paste frames
                master_frame.paste(frame_p0, (x_pos_p0, y_pos))
                master_frame.paste(frame_p1, (x_pos_p1, y_pos))
            
            master_frames.append(master_frame)
            if (frame_idx + 1) % 10 == 0:
                print(f"  Generated frame {frame_idx + 1}/{max_frames}")
        
        # Save master animation
        if not output_path.lower().endswith('.gif'):
            output_path = f"{os.path.splitext(output_path)[0]}_master_comparison.gif"
        
        master_frames[0].save(
            output_path,
            save_all=True,
            append_images=master_frames[1:],
            duration=duration,
            loop=0
        )
        
        print(f"\n🎉 Master comparison animation saved!")
        print(f"✓ Output: {output_path}")
        print(f"✓ Dimensions: {master_width}x{master_height}")
        print(f"✓ Grid: {num_rows} rows × {num_cols} columns")
        print(f"✓ Map types: {', '.join(valid_map_types)}")
        print(f"✓ Total frames: {len(master_frames)}")
        print(f"✓ Duration per frame: {duration}ms")

    def create_single_map_comparison(self, map_type: str, output_path: str,
                                   duration: int = 500) -> None:
        """Create a side-by-side comparison for a single map type."""
        print(f"Creating comparison for {map_type}...")
        
        # Get files for both partitions
        files_p0 = self.find_files_for_map_and_partition(map_type, 0)
        files_p1 = self.find_files_for_map_and_partition(map_type, 1)
        
        print(f"Partition 0: {len(files_p0)} files")
        print(f"Partition 1: {len(files_p1)} files")
        
        if not files_p0 and not files_p1:
            print(f"❌ No files found for {map_type}")
            return
        
        # Load and process images
        images_p0 = self.load_images_from_files(files_p0)
        images_p1 = self.load_images_from_files(files_p1)
        
        # Pad sequences to same length
        images_p0, images_p1 = self.pad_frame_sequences(images_p0, images_p1)
        
        # Resize to match
        all_images = images_p0 + images_p1
        all_images = self.resize_images_to_match(all_images)
        
        # Split back
        mid_point = len(all_images) // 2
        images_p0 = all_images[:mid_point]
        images_p1 = all_images[mid_point:]
        
        # Create side-by-side frames
        comparison_frames = []
        
        for img_p0, img_p1 in zip(images_p0, images_p1):
            # Create side-by-side frame
            combined_width = img_p0.width + img_p1.width
            combined_height = max(img_p0.height, img_p1.height)
            
            combined_frame = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
            combined_frame.paste(img_p0, (0, 0))
            combined_frame.paste(img_p1, (img_p0.width, 0))
            
            comparison_frames.append(combined_frame)
        
        # Save animation
        if not output_path.lower().endswith('.gif'):
            output_path = f"{os.path.splitext(output_path)[0]}_{map_type}_comparison.gif"
        
        comparison_frames[0].save(
            output_path,
            save_all=True,
            append_images=comparison_frames[1:],
            duration=duration,
            loop=0
        )
        
        print(f"✓ {map_type} comparison saved: {output_path}")
        print(f"✓ Total frames: {len(comparison_frames)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate side-by-side animation comparisons between partition 0 and 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/images -o master_comparison.gif
  %(prog)s /path/to/images -o comparison.gif -d 300 --maps target action traversability
  %(prog)s /path/to/images -o output.gif --single-map traversability
  %(prog)s /path/to/images --discover  # Discover available map types
        """
    )
    
    parser.add_argument("directory", help="Directory containing PNG files")
    parser.add_argument("-o", "--output", required=True, help="Output GIF file path")
    parser.add_argument("-d", "--duration", type=int, default=500,
                       help="GIF frame duration in milliseconds (default: 500)")
    parser.add_argument("--maps", nargs="+", 
                       help="Specific map types to include (default: auto-discover)")
    parser.add_argument("--single-map", type=str,
                       help="Create comparison for single map type only")
    parser.add_argument("--max-width", type=int, default=1200,
                       help="Maximum width for the output (default: 1200)")
    parser.add_argument("--font-size", type=int, 
                       help="Font size for labels (auto-calculated if not specified)")
    parser.add_argument("--no-labels", action="store_true",
                       help="Create animation without labels")
    parser.add_argument("--discover", action="store_true",
                       help="Discover and list available map types")
    
    args = parser.parse_args()
    
    try:
        generator = MasterMapAnimationGenerator(args.directory)
        
        # Discover available map types
        if args.discover:
            map_types = generator.discover_map_types()
            print(f"Available map types in {args.directory}:")
            print("-" * 50)
            for map_type in map_types:
                files_p0 = generator.find_files_for_map_and_partition(map_type, 0)
                files_p1 = generator.find_files_for_map_and_partition(map_type, 1)
                print(f"{map_type:20} - P0: {len(files_p0):3d} files, P1: {len(files_p1):3d} files")
            return
        
        # Single map comparison
        if args.single_map:
            generator.create_single_map_comparison(args.single_map, args.output, args.duration)
            return
        
        # Determine map types to process
        if args.maps:
            map_types = args.maps
        else:
            map_types = generator.discover_map_types()
            if not map_types:
                print("❌ No map types discovered. Use --discover to see available files.")
                sys.exit(1)
        
        # Create master comparison
        generator.create_master_comparison_animation(
            map_types, args.output, args.duration, args.max_width,
            args.font_size, args.no_labels
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()