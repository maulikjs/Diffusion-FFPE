import os
import numpy as np
import cupy as cp
import openslide
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm

def pad_to_size(image, target_size):
    """
    Pad image to exact target size, regardless of input dimensions
    
    Parameters:
    -----------
    image : PIL.Image
        Input image
    target_size : tuple
        (width, height) tuple of desired size
    """
    padded = Image.new('RGB', target_size, (0, 0, 0))
    padded.paste(image, (0, 0))
    return padded

def process_chunk(args):
    """
    Process a single chunk of the image ensuring fixed output dimensions.
    """
    input_file, output_folder, padding_info_dir, row, col, chunk_size, image_type, (width, height) = args
    
    try:
        if image_type == 'svs':
            slide = openslide.OpenSlide(input_file)
            
            # Calculate coordinates
            x = col * chunk_size
            y = row * chunk_size
            chunk_width = min(chunk_size, width - x)
            chunk_height = min(chunk_size, height - y)
            
            # Read region
            chunk = slide.read_region((x, y), 0, (chunk_width, chunk_height))
            chunk = chunk.convert('RGB')
            slide.close()
            
        else:  # TIFF
            with Image.open(input_file) as img:
                left = col * chunk_size
                top = row * chunk_size
                right = min(left + chunk_size, width)
                bottom = min(top + chunk_size, height)
                chunk = img.crop((left, top, right, bottom))
        
        # Store original size before padding
        original_size = chunk.size
        
        # Pad to exact chunk size (always 512x512 or whatever chunk_size is)
        padded_chunk = pad_to_size(chunk, (chunk_size, chunk_size))
        
        # Convert to NumPy array and transfer to GPU
        chunk_array = np.array(padded_chunk)
        gpu_chunk = cp.asarray(chunk_array)
        
        # Transfer back to CPU and save
        processed_chunk = cp.asnumpy(gpu_chunk)
        processed_image = Image.fromarray(processed_chunk)
        
        chunk_filename = os.path.join(output_folder, f"chunk_{row}_{col}.tiff")
        processed_image.save(chunk_filename, format="TIFF", compression='tiff_lzw')
        
        # Save dimensions in padding info directory
        meta_filename = os.path.join(padding_info_dir, f"chunk_{row}_{col}.txt")
        with open(meta_filename, 'w') as f:
            f.write(f"original_size: {original_size}\n")
            f.write(f"padded_size: {padded_chunk.size}\n")
            f.write(f"row: {row}\n")
            f.write(f"col: {col}\n")
            f.write(f"position: ({col * chunk_size}, {row * chunk_size})\n")
        
        return True
    except Exception as e:
        print(f"Error processing chunk at row {row}, col {col}: {e}")
        return False

def parallel_chunk_extraction(input_file, output_folder, padding_info_dir="./paddingInfo", chunk_size=512, num_workers=None):
    """
    Extract chunks from a large image in parallel using multiple CPU cores and GPU.
    Ensures all chunks are exactly chunk_size x chunk_size.
    """
    # Make chunk_size divisible by 4 to work with the model
    chunk_size = max(512, (chunk_size // 4) * 4)
    
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # Create output and padding info directories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(padding_info_dir, exist_ok=True)
    
    # Detect image type
    image_type = 'svs' if input_file.lower().endswith('.svs') else 'tiff'
    
    # Get image dimensions
    if image_type == 'svs':
        slide = openslide.OpenSlide(input_file)
        width, height = slide.level_dimensions[0]
        slide.close()
    else:
        with Image.open(input_file) as img:
            width, height = img.size
    
    # Calculate grid dimensions
    rows = int(np.ceil(height / chunk_size))
    cols = int(np.ceil(width / chunk_size))
    total_chunks = rows * cols
    
    print(f"Processing {total_chunks} chunks using {num_workers} workers...")
    print(f"Image dimensions: {width}x{height}")
    print(f"Fixed chunk size: {chunk_size}x{chunk_size}")
    print(f"Saving padding info to: {padding_info_dir}")
    
    # Prepare arguments for parallel processing
    args_list = [
        (input_file, output_folder, padding_info_dir, row, col, chunk_size, image_type, (width, height))
        for row in range(rows)
        for col in range(cols)
    ]
    
    # Process chunks in parallel with progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_chunk, args_list),
            total=len(args_list),
            desc="Extracting chunks"
        ))
    
    # Summary
    successful = sum(1 for r in results if r)
    failed = len(results) - successful
    
    print(f"\nExtraction complete:")
    print(f"Total chunks: {total_chunks}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    return {
        'total_chunks': total_chunks,
        'successful': successful,
        'failed': failed,
        'chunk_size': chunk_size,
        'width': width,
        'height': height,
        'workers': num_workers,
        'padding_info_dir': padding_info_dir
    }

if __name__ == '__main__':
    # Configure input/output
    input_file = "./testData/svstest.svs"  # or your TIFF file
    output_folder = "chunks"
    padding_info_dir = "./paddingInfo"
    
    # Optional: set specific number of workers
    num_workers = 10  # adjust based on your CPU cores
    
    # Increase PIL pixel limit
    Image.MAX_IMAGE_PIXELS = None
    
    try:
        result = parallel_chunk_extraction(
            input_file,
            output_folder,
            padding_info_dir=padding_info_dir,
            chunk_size=512,  # This will ensure 512x512 chunks
            num_workers=num_workers
        )
        
        print("\nProcessing Summary:")
        for key, value in result.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error: {e}")