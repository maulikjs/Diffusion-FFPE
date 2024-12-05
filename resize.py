from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = 10000000000  # 10 billion pixels

input_dir = "./"
output_dir = "./resizedData/"
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir(input_dir):
    if img_file.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path)
        img = img.resize((1024, 1024))  # Resize to 1024x1024 or smaller
        img.save(os.path.join(output_dir, img_file))

