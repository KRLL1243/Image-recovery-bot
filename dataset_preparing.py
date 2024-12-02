import os
from PIL import Image, ImageFilter
import random
from tqdm import tqdm

# Path to folder with original images
DATASET_PATH = "img_align_celeba"

# Path to the folder for processed images
OUTPUT_PATH = "prepared_dataset"
os.makedirs(OUTPUT_PATH, exist_ok=True)


# Function for adding blur
def add_blur(image: Image.Image):
    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 5)))

    return image


# Dataset preparation
def prepare_dataset(dataset_path, output_path, image_size=256):
    image_files = os.listdir(dataset_path)
    for image_file in tqdm(image_files, desc="Preparing dataset"):
        try:
            # Uploading an image
            image_path = os.path.join(dataset_path, image_file)
            original = Image.open(image_path).convert("RGB").resize((image_size, image_size))

            # Creating a blurred image
            blurred = add_blur(original)

            # Creating a paired image
            combined = Image.new("RGB", (image_size * 2, image_size))
            combined.paste(blurred, (0, 0))  # Blurred left
            combined.paste(original, (image_size, 0))  # Original right

            # Save
            combined.save(os.path.join(output_path, image_file))
        except Exception as e:
            print(f"Error with file {image_file}: {e}")


# Start preparation
prepare_dataset(DATASET_PATH, OUTPUT_PATH)