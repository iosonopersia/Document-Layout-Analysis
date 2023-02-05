import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils import get_config

if __name__ == '__main__':
    config = get_config()
    train_file = config.dataset.train_labels_file
    image_size = config.dataset.image_size
    filename = config.dataset.train_labels_file
    doc_categories = config.dataset.doc_categories

    # Load COCO dataset
    with open(filename, "r", encoding="utf-8") as f:
        coco_dataset = json.load(f)

    # Filter images by doc_category
    apply_filter: bool = (doc_categories is not None) and (len(doc_categories) > 0)
    images_set = [
        image['file_name']
        for image in coco_dataset['images']
        if not apply_filter or image['doc_category'] in doc_categories
        if image['precedence'] == 0
    ]

    print("Loading training images to compute mean and std across each color channel...")
    image_folder = os.path.abspath(config.dataset.images_dir)
    image_mean = np.zeros(3, dtype=np.float32)
    image_std = np.zeros(3, dtype=np.float32) # var[X] = E[X**2] - E[X]**2

    loop = tqdm(images_set, leave=True)
    loop.set_description(f"Processing training set images")
    loop.set_postfix(mean=0.0, std=0.0)

    num_processed_images = 0
    for image_filename in loop:
        image_path = os.path.join(image_folder, image_filename)

        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image, dtype=np.float32) # (H, W, C)
        image = image / 255.0

        image_mean += np.mean(image, axis=(0, 1))
        image_std += np.mean(np.square(image), axis=(0, 1))

        num_processed_images += 1

        # Update progress bar
        mean_to_display = image_mean / num_processed_images
        std_to_display = np.sqrt((image_std / num_processed_images) - np.square(mean_to_display))
        loop.set_postfix(mean=round(mean_to_display.mean(), 3), std=round(std_to_display.mean(), 3))

    mean_per_channel = image_mean / len(images_set)
    std_per_channel = np.sqrt((image_std / len(images_set)) - np.square(mean_per_channel))

    print(f'Mean: R={mean_per_channel[0]} G={mean_per_channel[1]} B={mean_per_channel[2]}')
    print(f'Std: R={std_per_channel[0]} G={std_per_channel[1]} B={std_per_channel[2]}')

    mean_std_filepath = os.path.abspath(config.dataset.mean_std_file)
    print(f'Saving mean and std to {mean_std_filepath}')
    with open(config.dataset.mean_std_file, 'wt', encoding='utf-8') as f:
        json.dump({
            'mean': mean_per_channel.tolist(),
            'std': std_per_channel.tolist(),
        }, f)
