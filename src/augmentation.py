import albumentations as A
import cv2

from utils import get_config

IMAGE_SIZE = get_config().dataset.image_size
SCALE = 1.2

train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * SCALE)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * SCALE),
            min_width=int(IMAGE_SIZE * SCALE),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.OneOf(
            [
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                    rotate=(-5, 5),
                    shear=(-10, 10),
                    p=0.5,
                    mode=cv2.BORDER_CONSTANT,
                    cval=(255, 255, 255)),
                A.Perspective(
                    scale=(0.05, 0.1),
                    p=0.5,
                    pad_mode=cv2.BORDER_CONSTANT,
                    pad_val=(255, 255, 255)),
            ],
            p=0.8),
        A.ToGray(p=0.2),
    ],
    bbox_params=A.BboxParams(format='coco', min_visibility=0.3, label_fields=['category_ids']),
    p=0.3, # ~30% of the images will be transformed
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE,
            min_width=IMAGE_SIZE,
            border_mode=cv2.BORDER_CONSTANT,
        ),
    ],
    bbox_params=A.BboxParams(format='coco', min_visibility=0.3, label_fields=['category_ids']),
)
