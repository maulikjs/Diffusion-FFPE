#!/usr/bin/env python3

import argparse
import json
import os
import numpy as np
import cupy as cp
import openslide
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import logging
import setup_logger 

logger = logging.getLogger('chunking')

def pad_to_size(image, target_size):
    """
    Pad image to exact target size, regardless of input dimensions
    """
    padded = Image.new('RGB', target_size, (0, 0, 0))
    padded.paste(image, (0, 0))
    return padded

def process_chunk(args):
    (input_file, output_folder, padding_info_dir, row, col,
     chunk_size, image_type, (width, height)) = args
    try:
        if image_type == 'svs':
            slide = openslide.OpenSlide(input_file)
            x = col * chunk_size
            y = row * chunk_size
            chunk_width = min(chunk_size, width - x)
            chunk_height = min(chunk_size, height - y)
            chunk = slide.read_region((x, y), 0, (chunk_width, chunk_height))
            chunk = chunk.convert('RGB')
            slide.close()
        else:
            with Image.open(input_file) as img:
                left = col * chunk_size
                top = row * chunk_size
                right = min(left + chunk_size, img.width)
                bottom = min(top + chunk_size, img.height)
                chunk = img.crop((left, top, right, bottom))

        original_size = chunk.size
        padded_chunk = pad_to_size(chunk, (chunk_size, chunk_size))

        # Convert to GPU array, then back to CPU (example usage of CuPy)
        chunk_array = np.array(padded_chunk)
        gpu_chunk = cp.asarray(chunk_array)
        processed_chunk = cp.asnumpy(gpu_chunk)

        processed_image = Image.fromarray(processed_chunk)
        chunk_filename = os.path.join(output_folder, f"chunk_{row}_{col}.tiff")
        processed_image.save(chunk_filename, format="TIFF", compression='tiff_lzw')

        # Save padding info
        meta_filename = os.path.join(padding_info_dir, f"chunk_{row}_{col}.txt")
        with open(meta_filename, 'w') as f:
            f.write(f"original_size: {original_size}\n")
            f.write(f"padded_size: {padded_chunk.size}\n")
            f.write(f"row: {row}\n")
            f.write(f"col: {col}\n")
            f.write(f"position: ({col * chunk_size}, {row * chunk_size})\n")

        return True
    except Exception as e:
        logger.exception(f"Error processing chunk at row {row}, col {col}: {e}")
        return False

def parallel_chunk_extraction(input_file, output_folder, padding_info_dir,
                              chunk_size=512, num_workers=None):
    """
    Extract chunks from a large SVS/TIFF image in parallel.
    """
    chunk_size = max(512, (chunk_size // 4) * 4)

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(padding_info_dir, exist_ok=True)

    image_type = 'svs' if input_file.lower().endswith('.svs') else 'tiff'

    if image_type == 'svs':
        slide = openslide.OpenSlide(input_file)
        width, height = slide.level_dimensions[0]
        slide.close()
    else:
        with Image.open(input_file) as img:
            width, height = img.size

    rows = int(np.ceil(height / chunk_size))
    cols = int(np.ceil(width / chunk_size))
    total_chunks = rows * cols

    logger.info(f"Processing {total_chunks} chunks with {num_workers} workers...")
    logger.info(f"Dimensions: {width}x{height}, chunk size: {chunk_size}")

    args_list = [
        (input_file, output_folder, padding_info_dir, r, c,
         chunk_size, image_type, (width, height))
        for r in range(rows) for c in range(cols)
    ]

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(process_chunk, args_list), total=len(args_list)):
            results.append(res)

    successful = sum(1 for r in results if r)
    failed = len(results) - successful

    return {
        'total_chunks': total_chunks,
        'successful': successful,
        'failed': failed
    }

def main():
    parser = argparse.ArgumentParser(description="Chunk a large SVS/TIFF into 512x512 tiles.")
    parser.add_argument('--input_file', required=True, help="Input SVS/TIFF file path")
    parser.add_argument('--output_folder', required=True, help="Folder to store chunk images")
    parser.add_argument('--padding_info_dir', default="./paddingInfo", help="Folder to store metadata")
    parser.add_argument('--chunk_size', type=int, default=512, help="Tile size, default 512")
    parser.add_argument('--num_workers', type=int, default=None, help="Number of parallel processes")
    parser.add_argument('--annotation_file', default="output_annotation.json",
                        help="Path to save the empty annotation JSON.")
    args = parser.parse_args()

    # Increase PIL pixel limit
    Image.MAX_IMAGE_PIXELS = None

    # Run chunk extraction
    results = parallel_chunk_extraction(
        args.input_file,
        args.output_folder,
        args.padding_info_dir,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers
    )

    # Create an empty annotation at (1,1)
    # HistomicsUI expects a JSON annotation of the form:
    # {
    #   "name": "AnnotationName",
    #   "elements": [ { "type": "point", "center": [x, y], ... } ]
    # }
    annotation = {
        "name": "Empty Annotation at (1,1)",
        "elements": [
            {
                "type": "point",
                "center": [1, 1],
                "points": [[1, 1]],
                "fillColor": "rgba(255, 0, 0, 1)",
                "lineColor": "rgba(0, 255, 0, 1)",
            }
        ]
    }

    with open(args.annotation_file, 'w') as af:
        json.dump(annotation, af)

    # Print out final results so we can see them in the job log
    logger.info("Results:", results)
    logger.info("Annotation saved to:", args.annotation_file)

if __name__ == "__main__":
    main()
