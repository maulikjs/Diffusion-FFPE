import os
import tifffile
import numpy as np
from PIL import Image

def stitch_chunks_to_tiff_big(input_folder, output_file, original_width, original_height, chunk_size=512):
    chunk_files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('_')[1].split('.')[0]))
    rows = original_height // chunk_size
    cols = original_width // chunk_size
    stitched_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)  # RGB image

    chunk_count = 0
    for row in range(rows):
        for col in range(cols):
            top = row * chunk_size
            left = col * chunk_size
            if chunk_count < len(chunk_files):
                chunk_path = os.path.join(input_folder, chunk_files[chunk_count])
                chunk = np.array(Image.open(chunk_path))
                stitched_image[top:top+chunk.shape[0], left:left+chunk.shape[1], :] = chunk
                chunk_count += 1

    # Save the stitched image with JPEG compression
    tifffile.imwrite(output_file, stitched_image, bigtiff=True, compression='lzw')
    print(f"Stitched image saved as {output_file}")

# Define the input folder and output file
input_folder = "convertedOutput"
output_file = "stitched_image_compressed.tiff"
original_width = 94208
original_height = 75264

# Stitch the chunks back together
stitch_chunks_to_tiff_big(input_folder, output_file, original_width, original_height)
