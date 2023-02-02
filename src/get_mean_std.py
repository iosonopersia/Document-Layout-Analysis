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
    # Load COCO dataset
    with open(filename, "r", encoding="utf-8") as f:
        coco_dataset = json.load(f)

    # Filter images by doc_category
    images_set = [
        image['file_name']
        for image in coco_dataset['images']
        if image['doc_category'] in ['scientific_articles']
    ]

    print("Loading training images to compute mean and std across each color channel...")
    image_folder = os.path.abspath(config.dataset.images_dir)
    images = []
    for image_filename in tqdm(images_set):
        image_path = os.path.join(image_folder, image_filename)

        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image, dtype=np.float32)
        image = image / 255.0

        images.append(image)

    images = np.stack(images, axis=0) # (N, H, W, C)
    mean_per_channel = np.mean(images, axis=(0, 1, 2))
    std_per_channel = np.std(images, axis=(0, 1, 2))

    print(f'Mean: R={mean_per_channel[0]} G={mean_per_channel[1]} B={mean_per_channel[2]}')
    print(f'Std: R={std_per_channel[0]} G={std_per_channel[1]} B={std_per_channel[2]}')

