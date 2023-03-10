import json
import os
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BeitFeatureExtractor, logging

from augmentation import test_transforms, train_transforms
from coco import COCOAnnotation, COCOBoundingBox
from utils import get_anchors_dict, get_config, get_mean_std

logging.set_verbosity_error()


class COCODataset(Dataset):
    def __init__(
        self,
        coco_file_path,
        images_dir,
        anchors_dict = {},
        doc_categories=None,
        transform=None,
        feature_extractor=None,
    ):
        if anchors_dict is None or len(anchors_dict) == 0:
            raise ValueError("anchors must be a non-empty dictionary.")

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
        self.C: int = len(self.id_to_categories)

        # Image ID to filename (keeps only images with doc_category in filter_doc_categories)
        apply_filter: bool = (doc_categories is not None) and (len(doc_categories) > 0)
        self.id_to_image_filename: dict[int, str] = {
            image['id']: image['file_name']
            for image in coco_dataset['images']
            if not apply_filter or image['doc_category'] in doc_categories
            if image['precedence'] == 0
        }

        # List of image IDs
        self.images: list[int] = list(self.id_to_image_filename.keys())

        # Annotation ID to COCOAnnotation object
        images_set: set[int] = set(self.id_to_image_filename.keys()) # Convert to set for faster lookup
        images_with_annotations: set[int] = set() # Images with at least one annotation

        self.annotations: dict[int, COCOAnnotation] = dict()
        for annotation in coco_dataset['annotations']:
            if annotation['image_id'] in images_set and \
               annotation['iscrowd'] == 0 and \
               annotation['precedence'] == 0:
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

        # Anchors (here we assume same num of anchors per scale)
        self.anchors: torch.Tensor = torch.stack([anchors_dict[s] for s in self.S], dim=0) # (4, 3, 2)
        self.num_anchors_per_scale: int = self.anchors.shape[1]
        self.ignore_iou_thresh: float = 0.5

        self.transform = transform

        self.images = list(images_with_annotations)
        self.dataset_size = len(self.images)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        image_id = self.images[index]

        # Load annotations
        annotations = self.annotations_for_image[image_id]
        bboxes: list[COCOBoundingBox] = [annotation.bbox for annotation in annotations]
        category_ids: list[int] = [annotation.category_id for annotation in annotations]

        # Load image tensor
        image_filename = self.id_to_image_filename[image_id]
        image_path = os.path.join(self.img_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        image = np.asarray(image, dtype=np.uint8)

        # Apply augmentations
        if self.transform is not None:
            bboxes: list[list[float]] = [bbox.to_list() for bbox in bboxes]

            augmentations = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
            image: torch.Tensor = augmentations["image"] # Tensor of shape (C, H, W) -> (3, 1025, 1025)
            bboxes: list[list[float]] = augmentations["bboxes"] # List of bounding boxes (x, y, w, h)
            category_ids: list[int] = augmentations["category_ids"] # List of category IDs

            # Convert bboxes back to COCOBoundingBox objects
            # (sanity checks are performed on transformed bboxes)
            bboxes: list[COCOBoundingBox] = [
                COCOBoundingBox(*bbox, max_width=orig_width, max_height=orig_height)
                for bbox in bboxes
            ]

        # from visualization import visualize
        # visualize(image.permute(1, 2, 0).numpy(), bboxes, category_ids, self.id_to_categories)

        # Create targets
        targets = {S: torch.zeros((self.num_anchors_per_scale, S, S, 5+self.C), dtype=torch.float32) for S in self.S}
        for bbox, class_label in zip(bboxes, category_ids):
            bbox = bbox.to_normalized()
            x, y, width, height = bbox.x_center, bbox.y_center, bbox.w, bbox.h # Normalized to [0, 1]

            gt_shape: torch.Tensor = torch.tensor([width, height], dtype=torch.float32) # (2,)

            jaccard_similarities = self._jaccard_similarity(gt_shape, self.anchors) # (4, 3) = (num_scales, num_anchors_per_scale)
            sorted_anchor_indices = jaccard_similarities.argsort(dim=-1, descending=True)

            for scale_idx, S in enumerate(self.S):
                j, i = int(S * x), int(S * y) # Indices of the cell in which the center of the GT-bbox falls
                dx, dy = (S * x) - j, (S * y) - i # Offsets of the center of the GT-bbox from the top-left corner of the cell

                is_anchor_assigned: bool = False
                for anchor_idx in sorted_anchor_indices[scale_idx, :].tolist():
                    is_anchor_available: bool = targets[S][anchor_idx, i, j, 0] == 0 # -1 means ignore, 1 means object

                    if is_anchor_available:
                        if not is_anchor_assigned:
                            target_tensor = torch.zeros((5+self.C), dtype=torch.float32)
                            target_tensor[0] = 1 # Objectness score
                            target_tensor[1:5] = torch.tensor([dx, dy, S*width, S*height], dtype=torch.float32) # Bbox coords
                            target_tensor[5+class_label-1] = 1 # Class label (subtract 1 to make class labels start from 0)

                            targets[S][anchor_idx, i, j, :] = target_tensor
                            is_anchor_assigned = True
                        else:
                            # An anchor for the current scale has already been assigned to a bbox
                            if jaccard_similarities[scale_idx, anchor_idx] > self.ignore_iou_thresh:
                                targets[S][anchor_idx, i, j, [0]] = -1

        # Extract features
        image: torch.Tensor = self.feature_extractor(image, return_tensors='pt').pixel_values # (1, C, H, W)
        image = image.squeeze(dim=0) # (C, H, W)

        return {'image_id': image_id, 'image': image, 'target': targets}

    def _jaccard_similarity(self, wh_1: torch.Tensor, wh_2: torch.Tensor) -> torch.Tensor:
        w1 = wh_1[..., 0]
        h1 = wh_1[..., 1]
        w2 = wh_2[..., 0]
        h2 = wh_2[..., 1]
        intersection = torch.min(w1, w2) * torch.min(h1, h2)
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

        # Override default mean and std values with those that are suitable for the dataset
        mean_std_path = os.path.abspath(config.dataset.mean_std_file)
        mean, std = get_mean_std(mean_std_path)
        feature_extractor.do_normalize = True
        feature_extractor.image_mean = mean
        feature_extractor.image_std = std

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
        doc_categories=dataset_cfg.doc_categories,
        transform=train_transforms if apply_transforms else test_transforms,
        feature_extractor=feature_extractor
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        **config_dataloader
    )

    return dataloader
