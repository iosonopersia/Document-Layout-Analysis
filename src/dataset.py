import json
import os
import warnings

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BeitFeatureExtractor, logging

from augmentation import test_transforms, train_transforms
from coco import COCOAnnotation, COCOBoundingBox
from utils import get_anchors_dict, get_config

logging.set_verbosity_error()


class COCODataset(Dataset):
    def __init__(
        self,
        coco_file_path,
        images_dir,
        anchors_dict = {},
        filter_doc_categories={},
        transform=None,
        feature_extractor=None,
    ):
        if anchors_dict is None or len(anchors_dict) == 0:
            raise ValueError("anchors must be a non-empty dictionary.")

        if filter_doc_categories is None or len(filter_doc_categories) == 0:
            raise ValueError("filter_doc_categories must be a non-empty set.")

        if feature_extractor is None:
            raise ValueError("feature_extractor must be provided.")

        self.feature_extractor = feature_extractor

        # Load COCO dataset
        with open(coco_file_path, "r", encoding="utf-8") as f:
            coco_dataset = json.load(f)

        # Category ID to name
        self.id_to_categories: dict[int, str] = {
            category['id']: category['name']
            for category in coco_dataset['categories']}
        self.num_categories: int = len(self.id_to_categories)

        # Image ID to filename (keeps only images with doc_category in filter_doc_categories)
        self.id_to_image_filename: dict[int, str] = {
            image['id']: image['file_name']
            for image in coco_dataset['images']
            if image['doc_category'] in filter_doc_categories}

        # List of image IDs
        self.images: list[int] = list(self.id_to_image_filename.keys())

        # Annotation ID to COCOAnnotation object
        images_set: set[int] = set(self.id_to_image_filename.keys()) # Convert to set for faster lookup
        images_with_annotations: set[int] = set() # Images with at least one annotation

        self.annotations: dict[int, COCOAnnotation] = dict()
        for annotation in coco_dataset['annotations']:
            if annotation['image_id'] in images_set:
                try:
                    bbox = COCOBoundingBox(*annotation['bbox'], max_width=1025, max_height=1025)
                    self.annotations[annotation['id']] = COCOAnnotation(
                        annotation['id'],
                        annotation['image_id'],
                        annotation['category_id'],
                        bbox)
                    images_with_annotations.add(annotation['image_id'])
                except ValueError as e:
                    annotation_id = annotation['id']
                    image_id = annotation['image_id']
                    warnings.warn(f"[Annotation {annotation_id}, Image {image_id}]: {e}", UserWarning)

        # Image ID to list of annotation IDs
        self.annotations_for_image: dict[int, list[COCOAnnotation]] = {}
        for annotation in self.annotations.values():
            image_id = annotation.image_id
            if image_id in self.annotations_for_image:
                self.annotations_for_image[image_id].append(annotation)
            else:
                self.annotations_for_image[image_id] = [annotation]

        # Directory containing the images
        self.img_dir: str = os.path.abspath(images_dir)
        if not os.path.isdir(self.img_dir):
            raise ValueError(f"images_dir={images_dir} is not a valid directory.")

        # Grid sizes
        self.S: list[int] = list(anchors_dict.keys()) # Grid sizes
        self.num_scales: int = len(self.S)

        # Anchors
        self.anchor_shapes: dict[int, list[float]] = anchors_dict
        self.num_anchors_per_scale: int = len(anchors_dict[self.S[0]]) # Here we assume same num of anchors per scale
        self.ignore_iou_thresh: float = 0.5

        default_transform = A.Compose(
            [A.ToFloat(max_value=255), A.pytorch.ToTensorV2()],
            bbox_params=A.BboxParams(format="coco", min_visibility=0.3, label_fields=['category_ids']))

        self.transform = transform if transform is not None else default_transform

        self.images = list(images_with_annotations)
        self.dataset_size = len(self.images)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        image_id = self.images[index]

        # Load annotations
        annotations = self.annotations_for_image[image_id]

        # Load image tensor
        image_filename = self.id_to_image_filename[image_id]
        image_path = os.path.join(self.img_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        image = np.asarray(image, dtype=np.uint8)

        # Apply augmentations
        bboxes = [annotation.bbox.to_list() for annotation in annotations]
        category_ids = [annotation.category_id for annotation in annotations]

        augmentations = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
        image: torch.Tensor = augmentations["image"] # Tensor of shape (C, H, W) -> (3, 1025, 1025)
        bboxes: list[list[float]] = augmentations["bboxes"] # List of bounding boxes (x, y, w, h)
        category_ids: list[int] = augmentations["category_ids"] # List of category IDs

        # Convert bboxes back to COCOBoundingBox objects (sanity checks are performed on transformed bboxes)
        bboxes: list[COCOBoundingBox] = [
            COCOBoundingBox(*bbox, max_width=orig_width, max_height=orig_height)
            for bbox in bboxes
        ]

        # from visualization import visualize
        # visualize(image.permute(1, 2, 0).numpy(), bboxes, category_ids, self.id_to_categories)

        # Create targets
        # 6 = 1 objectness score + 4 bbox coords + 1 class label (p_obj, x, y, w, h, class)
        targets = {S: torch.zeros((self.num_anchors_per_scale, S, S, 6), dtype=torch.float32) for S in self.S}
        for bbox, class_label in zip(bboxes, category_ids):
            bbox = bbox.to_normalized()
            x, y, width, height = bbox.x_center, bbox.y_center, bbox.w, bbox.h # Normalized to [0, 1]

            for S in self.S:
                i, j = int(S * y), int(S * x) # Indices of the cell in which the center of the GT-bbox falls

                anchors = self.anchor_shapes[S] # Anchors for this scale

                # Sort anchors by shape similarity with ground truth box (descending order)
                anchor_similarities = {i: self._shape_similarity(bbox.w, bbox.h, *anchor) for i, anchor in enumerate(anchors)}
                sorted_anchor_indices = sorted(anchor_similarities.keys(), key=lambda i: anchor_similarities[i], reverse=True)

                is_anchor_assigned = False
                for anchor_idx in sorted_anchor_indices:
                    anchor_p_obj = targets[S][anchor_idx, i, j, 0].item()
                    is_anchor_available = int(anchor_p_obj) not in (-1, 1) # -1 means ignore, 1 means object

                    if is_anchor_available:
                        if not is_anchor_assigned:
                            # X and Y coordinates of the center of the bbox, scaled to the size of the current feature map
                            x_scaled, y_scaled = x * S, y * S

                            # Width and height of the bbox, scaled to the size of the current feature map
                            width_scaled, height_scaled = width * S, height * S,

                            targets[S][anchor_idx, i, j, :] = torch.tensor(
                                [1, x_scaled, y_scaled, width_scaled, height_scaled, (class_label - 1)], # Subtract 1 to make class labels start from 0
                                dtype=torch.float32)

                            is_anchor_assigned = True
                        else:
                            # An anchor for the current scale has already been assigned to a bbox
                            if anchor_similarities[anchor_idx] > self.ignore_iou_thresh:
                                targets[S][anchor_idx, i, j, [0]] = -1

        # Extract features
        image = self.feature_extractor(image, return_tensors='pt').pixel_values
        image = image.squeeze(dim=0) # (C, H, W)

        return {'image': image, 'target': targets}

    def _shape_similarity(self, w1: float, h1: float, w2: float, h2: float) -> float:
        intersection = min(w1, w2) * min(h1, h2)
        union = (w1 * h1) + (w2 * h2) - intersection
        return intersection / union


feature_extractor: BeitFeatureExtractor = None

def get_dataloader(split: str = "train") -> tuple[COCODataset, torch.utils.data.DataLoader]:
    config = get_config()

    # Load feature extractor only once
    global feature_extractor
    if feature_extractor is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_cfg = config.model
            feature_extractor = BeitFeatureExtractor.from_pretrained(model_cfg.backbone)

    # Select config for the required split
    if split in ["train", "val", "test"]:
        config_dataloader = config.dataloader[split]
    else:
        raise ValueError("Invalid split")

    # Create dataset and dataloader
    dataset_cfg = config.dataset
    apply_transforms = split == "train" and dataset_cfg.apply_augmentations
    dataset = COCODataset(
        coco_file_path=dataset_cfg[split + "_labels_file"],
        images_dir=dataset_cfg.images_dir,
        anchors_dict=get_anchors_dict(dataset_cfg.anchors_file),
        filter_doc_categories=dataset_cfg.doc_categories,
        transform=train_transforms if apply_transforms else test_transforms,
        feature_extractor=feature_extractor
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        **config_dataloader
    )

    return dataloader
