import os
import tifffile
import numpy as np
from PIL import Image

def convert_and_match_specs(input_path, output_path, ref_path):
    """
    Converts and matches the specifications of a reference TIFF image.
    
    Parameters:
    - input_path: Path to the model output TIFF image.
    - output_path: Path to save the converted image.
    - ref_path: Path to the reference TIFF image with desired specs.
    """
    # Read the reference image to extract specifications
    with tifffile.TiffFile(ref_path) as ref_tif:
        ref_metadata = ref_tif.pages[0]
        ref_width, ref_height = ref_metadata.imagewidth, ref_metadata.imagelength
        ref_compression = ref_metadata.compression
        ref_rows_per_strip = ref_metadata.rowsperstrip

    # Read the input image
    with tifffile.TiffFile(input_path) as input_tif:
        input_image = input_tif.asarray()

    # Resize the input image to match the reference dimensions
    resized_image = np.array(
        Image.fromarray(input_image).resize((ref_width, ref_height), resample=Image.LANCZOS)
    )

    # Save the resized image with matching specifications
    tifffile.imwrite(
        output_path,
        resized_image,
        compression="jpeg" if ref_compression == 7 else None,
        photometric="rgb",
        rowsperstrip=ref_rows_per_strip,
    )
    print(f"Converted image saved to: {output_path}")

def batch_convert(input_folder, output_folder, ref_file):
    """
    Processes all images in the input folder and saves them to the output folder.
    
    Parameters:
    - input_folder: Folder containing images to be converted.
    - output_folder: Folder to save converted images.
    - ref_file: Path to a reference TIFF image with desired specs.
    """
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.tiff', '.tif')):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            try:
                convert_and_match_specs(input_path, output_path, ref_file)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Folders and reference file
input_folder = "./output"  # Folder containing model output images
output_folder = "./convertedOutput"  # Folder to save converted images
reference_file = "./chunks/chunk_1.tiff"  # Reference image for specs

# Batch convert images
batch_convert(input_folder, output_folder, reference_file)
