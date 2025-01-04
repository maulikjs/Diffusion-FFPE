import pyvips

# Open the large TIFF in streaming mode
image = pyvips.Image.new_from_file("stitched_output.tiff", access="sequential")

# Resize by a factor of 0.1 (1/10th)
# The `resize` method scales the image dimensions by the given factor.
resized = image.resize(0.1)

# Write out the resized image as a TIFF
resized.write_to_file("stitched_output_resized_input.tiff")
