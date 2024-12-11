import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import re
import tifffile

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

def stitch_image(chunks_dir, padding_info_dir, output_path, tile_size=240):
    """Stitch chunks into a single tiled TIFF."""
    
    # Get dimensions
    print("Calculating image dimensions...")
    width, height = get_image_dimensions(padding_info_dir)
    print(f"Full image size: {width}x{height}")
    
    # Create empty output array
    print("Creating output array...")
    output = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Process all chunks
    print("Processing chunks...")
    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.tiff')]
    
    for chunk_file in tqdm(chunk_files):
        # Get padding info
        info_file = os.path.splitext(chunk_file)[0] + '.txt'
        info_path = os.path.join(padding_info_dir, info_file)
        
        if not os.path.exists(info_path):
            continue
            
        # Load chunk and info
        info = parse_padding_info(info_path)
        chunk = np.array(Image.open(os.path.join(chunks_dir, chunk_file)))
        
        # Crop if needed
        if chunk.shape[:2] != (info['original_height'], info['original_width']):
            chunk = chunk[:info['original_height'], :info['original_width']]
        
        # Place in output array
        output[info['y']:info['y']+info['original_height'], 
               info['x']:info['x']+info['original_width']] = chunk
        
    # Save as single tiled TIFF
    print(f"Saving tiled TIFF to {output_path}")
    tifffile.imwrite(output_path,
                     output,
                     bigtiff=True,  # Enable BigTIFF format
                     tile=(tile_size, tile_size),
                     compression='zlib',
                     photometric='rgb',
                     metadata={'ImageDescription': f'Processed from {len(chunk_files)} chunks'})
    
    return {
        'width': width,
        'height': height,
        'chunks_processed': len(chunk_files),
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
            tile_size=240
        )
        
        print("\nStitching Summary:")
        for key, value in result.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")