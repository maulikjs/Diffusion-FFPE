from PIL import Image

# Open the TIFF image
input_path = 'stitched_image_compressed.tiff'
output_path = 'resized_image.tiff'
Image.MAX_IMAGE_PIXELS = 10000000000  # 10 billion pixels

# Open the image
with Image.open(input_path) as img:
    # Resize the image 
    # Using Image.LANCZOS (formerly Image.ANTIALIAS) for high-quality downsampling
    resized_img = img.resize((11776*2, 9408*2), Image.LANCZOS)
    
    # Save the resized image
    # Preserve the compression and other metadata if possible
    resized_img.save(output_path, compression='tiff_lzw')

print(f"Image resized and saved to {output_path}")