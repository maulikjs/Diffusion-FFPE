from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = 10000000000  # 10 billion pixels

def split_tiff_to_chunks(input_file, output_folder, chunk_size=512):
    # Open the large TIFF image
    with Image.open(input_file) as img:
        width, height = img.size
        print(f"Image dimensions: {width}x{height}")

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Loop through the image and create chunks
        chunk_count = 0
        for top in range(0, height, chunk_size):
            for left in range(0, width, chunk_size):
                # Define the bounding box for the current chunk
                right = min(left + chunk_size, width)
                bottom = min(top + chunk_size, height)
                box = (left, top, right, bottom)

                # Crop the chunk from the image
                chunk = img.crop(box)

                # Save the chunk to the output folder
                chunk_name = f"chunk_{chunk_count}.tiff"
                chunk.save(os.path.join(output_folder, chunk_name), format="TIFF")
                chunk_count += 1

        print(f"Total chunks created: {chunk_count}")

# Define input TIFF file and output folder
input_file = "./patient_198_node_1.tif"  # Replace with your TIFF file path
output_folder = "chunks"  # Folder to save the chunks

# Split the TIFF file into chunks of 512x512
split_tiff_to_chunks(input_file, output_folder)

