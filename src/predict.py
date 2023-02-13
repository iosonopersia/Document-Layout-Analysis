import os
import warnings
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from transformers import BeitFeatureExtractor

from coco import COCOBoundingBox
from metrics.bboxes_extraction import predictions_to_bboxes
from metrics.nms import non_max_suppression
from model import DocumentObjectDetector
from tools.checkpoint_handler import CheckpointHandler
from utils import get_anchors_dict, get_config, get_mean_std
from visualization import (BLUE, CYAN, GREEN, MAGENTA, MAROON, OLIVE, PURPLE,
                           RED, SILVER, TEAL, YELLOW, visualize)


def run_inference(image_path: str):
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found!")

    image = Image.open(image_path).convert('RGB')
    image = np.asarray(image, dtype=np.uint8) # (H, W, C)
    image_patches: torch.Tensor = feature_extractor(image, return_tensors='pt').pixel_values # (1, C, H, W)

    # ===========MODEL===============
    model = DocumentObjectDetector(NUM_CLASSES, model_cfg)
    model = model.to(DEVICE)
    checkpoint_handler.load_for_testing(test_cfg.checkpoint_path, model)

    # ===========INFERENCE===========
    with torch.inference_mode():
        predictions = model(image_patches.to(DEVICE))

    for S in predictions.keys():
        predictions[S] = predictions[S].cpu()

    # ===========================
    # Get bboxes from predictions
    # ===========================
    image_ids = torch.zeros(1, dtype=torch.int64)
    bboxes = predictions_to_bboxes(predictions, image_ids, SCALED_ANCHORS) # (#PRED_BOXES, 7)

    nms_boxes = non_max_suppression(
        bboxes[0, ...],
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold
    )

    # Convert from YOLO format to Pascal COCO format
    nms_boxes[:, [3, 4]] -= nms_boxes[:, [5, 6]] / 2
    nms_boxes[:, [3, 4, 5, 6]] = nms_boxes[:, [3, 4, 5, 6]].clamp(min=0, max=1025)

    pred_bboxes = [COCOBoundingBox(*nms_boxes[i, [3, 4, 5, 6]].tolist(), max_width=1025, max_height=1025) for i in range(len(nms_boxes))]
    pred_classes = (nms_boxes[:, 1] + 1).tolist()

    visualize(image, pred_bboxes, pred_classes, id_to_categories, id_to_colour, is_normalized=False)


if __name__ == "__main__":
    #============DEVICE===============
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Using device {DEVICE}]")

    # ===========CONFIG===============
    config = get_config()
    dataset_cfg = config.dataset
    model_cfg = config.model
    test_cfg = config.test
    iou_threshold = test_cfg.non_max_suppression.min_iou
    confidence_threshold = test_cfg.non_max_suppression.min_confidence

    # =============TOOLS==============
    checkpoint_handler = CheckpointHandler(config.checkpoint)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_cfg = config.model
        feature_extractor = BeitFeatureExtractor.from_pretrained(model_cfg.backbone)

        # Override default mean and std values with those that are suitable for the dataset
        mean_std_path = os.path.abspath(dataset_cfg.mean_std_file)
        mean, std = get_mean_std(mean_std_path)
        feature_extractor.do_normalize = True
        feature_extractor.image_mean = mean
        feature_extractor.image_std = std

    # ===========DATASET==============
    NUM_CLASSES = dataset_cfg.num_classes
    ANCHORS_DICT = get_anchors_dict(dataset_cfg.anchors_file)
    SCALED_ANCHORS = OrderedDict({
        size: anchors * size
        for size, anchors in ANCHORS_DICT.items()
    })

    id_to_categories = {
        1: "Caption",
        2: "Footnote",
        3: "Formula",
        4: "List-item",
        5: "Page-footer",
        6: "Page-header",
        7: "Picture",
        8: "Section-header",
        9: "Table",
        10: "Text",
        11: "Title",
    }

    id_to_colour = {
        1: RED,
        2: GREEN,
        3: BLUE,
        4: YELLOW,
        5: CYAN,
        6: MAGENTA,
        7: SILVER,
        8: MAROON,
        9: OLIVE,
        10: PURPLE,
        11: TEAL
    }

    # Example of test sample (scientific article)
    # run_inference(image_path="C:\\DocLayNet_core\\PNG\\8cd810ceca8a4a2776951582b93c65f1a17fc0d0b52e6194fc71fcc1e42ba711.png")

    # Example of test sample (financial report)
    # run_inference(image_path="C:\\DocLayNet_core\\PNG\\e918f494e45e8420892b22c95e961e1aee26d9e2c757479f1616bc4cc7599a0d.png")
