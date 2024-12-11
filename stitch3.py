import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
import math
import tifffile
import warnings

def parse_padding_info(file_path):
    """Parse the padding info file to get original dimensions and position."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    info = {}
    for line in lines:
        if 'original_size' in line:
            match = re.search(r'\((\d+), (\d+)\)', line)
            if match:
                info['original_width'] = int(match.group(1))
                info['original_height'] = int(match.group(2))
        elif 'position' in line:
            match = re.search(r'\((\d+), (\d+)\)', line)
            if match:
                info['x'] = int(match.group(1))
                info['y'] = int(match.group(2))
    return info

def get_image_dimensions(padding_info_dir):
    """Calculate full image dimensions from padding info."""
    max_x = 0
    max_y = 0
    
    for info_file in os.listdir(padding_info_dir):
        if not info_file.endswith('.txt'):
            continue
        
        info = parse_padding_info(os.path.join(padding_info_dir, info_file))
        max_x = max(max_x, info['x'] + info['original_width'])
        max_y = max(max_y, info['y'] + info['original_height'])
    
    return max_x, max_y

def create_chunk_map(chunks_dir, padding_info_dir):
    """Create a mapping of chunk positions and their files."""
    chunk_map = {}
    
    for chunk_file in os.listdir(chunks_dir):
        if not chunk_file.endswith('.tiff'):
            continue
            
        info_file = os.path.splitext(chunk_file)[0] + '.txt'
        info_path = os.path.join(padding_info_dir, info_file)
        
        if os.path.exists(info_path):
            info = parse_padding_info(info_path)
            chunk_map[chunk_file] = {
                'x': info['x'],
                'y': info['y'],
                'width': info['original_width'],
                'height': info['original_height']
            }
    
    return chunk_map

def process_section(chunks_dir, chunk_map, section_bounds, section_index, max_width):
    """Process a section of the image using tiled approach."""
    left, top, right, bottom = section_bounds
    section_height = bottom - top
    
    # Create empty section array
    section = np.zeros((section_height, max_width, 3), dtype=np.uint8)
    
    # Find chunks that intersect with this section
    for chunk_file, info in chunk_map.items():
        chunk_right = info['x'] + info['width']
        chunk_bottom = info['y'] + info['height']
        
        # Check if chunk intersects with this section
        if (info['y'] < bottom and chunk_bottom > top):
            try:
                # Load chunk
                chunk_path = os.path.join(chunks_dir, chunk_file)
                chunk = np.array(Image.open(chunk_path))
                
                # Calculate intersection
                chunk_top = max(0, top - info['y'])
                chunk_bottom = min(chunk.shape[0], bottom - info['y'])
                
                # Calculate destination
                dest_top = max(0, info['y'] - top)
                dest_bottom = min(section_height, dest_top + (chunk_bottom - chunk_top))
                dest_left = info['x']
                dest_right = min(max_width, info['x'] + chunk.shape[1])
                
                if dest_bottom > dest_top and dest_right > dest_left:
                    section[dest_top:dest_bottom, dest_left:dest_right] = \
                        chunk[chunk_top:chunk_bottom, 0:(dest_right-dest_left)]
                
            except Exception as e:
                print(f"Error processing chunk {chunk_file}: {str(e)}")
                continue
    
    return section

def stitch_image(chunks_dir, padding_info_dir, output_path, section_height=5000, tile_size=240):
    """
    Stitch chunks using tiled output with compression.
    """
    # Get dimensions and create chunk map
    print("Calculating image dimensions...")
    width, height = get_image_dimensions(padding_info_dir)
    print(f"Full image size: {width}x{height}")
    
    print("Creating chunk mapping...")
    chunk_map = create_chunk_map(chunks_dir, padding_info_dir)
    
    # Calculate number of sections
    num_sections = math.ceil(height / section_height)
    print(f"Processing image in {num_sections} sections...")
    
    # Configure TIFF options
    options = dict(
        tile=(tile_size, tile_size),  # Set tile size
        photometric='rgb',
        compression='zlib',           # Use zlib compression
        metadata={
            'ImageDescription': f'Processed from {len(chunk_map)} chunks'
        }
    )
    
    print(f"Creating tiled output with {tile_size}x{tile_size} tiles using zlib compression...")
    
    # Open output file
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        # Process each section
        for i in tqdm(range(num_sections), desc="Processing sections"):
            top = i * section_height
            bottom = min(top + section_height, height)
            
            # Process section
            section = process_section(
                chunks_dir,
                chunk_map,
                (0, top, width, bottom),
                i,
                width
            )
            
            # Write section with tiles
            tif.write(section, **options)
            
            # Clean up
            del section
            import gc
            gc.collect()
    
    return {
        'width': width,
        'height': height,
        'chunks_processed': len(chunk_map),
        'sections_processed': num_sections,
        'tile_size': tile_size
    }

if __name__ == '__main__':
    # Configure paths
    chunks_dir = "output"
    padding_info_dir = "./paddingInfo"
    output_path = "stitched_output.tiff"
    
    try:
        import tifffile
    except ImportError:
        print("Installing required package tifffile...")
        import subprocess
        subprocess.check_call(["pip", "install", "tifffile"])
        import tifffile
    
    try:
        result = stitch_image(
            chunks_dir, 
            padding_info_dir, 
            output_path,
            section_height=5000,    # Process in 5000-pixel-high sections
            tile_size=240          # Match original tile size
        )
        
        print("\nStitching Summary:")
        for key, value in result.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")