import pickle
from collections import OrderedDict

import torch

from dataset import get_dataloader
from metrics.bboxes_extraction import get_evaluation_bboxes
from metrics.map import mean_average_precision
from model import DocumentObjectDetector
from tools.checkpoint_handler import CheckpointHandler
from utils import get_anchors_dict, get_config


def get_bboxes_from_test_set():
    # ========== DATASET=============
    eval_loader = get_dataloader("test")

    # ===========MODEL===============
    model = DocumentObjectDetector(NUM_CLASSES, model_cfg)
    model = model.to(DEVICE)
    checkpoint_handler.load_for_testing(test_cfg.checkpoint_path, model)

    # ============TEST===============
    print("Extracting bboxes from model predictions on the test set...")
    pred_boxes, true_boxes = get_evaluation_bboxes(
        eval_loader,
        model,
        iou_threshold=test_cfg.non_max_suppression.min_iou,
        anchors=SCALED_ANCHORS,
        confidence_threshold=test_cfg.non_max_suppression.min_confidence,
    )
    with open("data/pred_boxes.pkl", "wb") as f:
        pickle.dump(pred_boxes, f)
    with open("data/true_boxes.pkl", "wb") as f:
        pickle.dump(true_boxes, f)


def test_metrics():
    with open("data/pred_boxes.pkl", "rb") as f:
        pred_boxes = pickle.load(f)
    with open("data/true_boxes.pkl", "rb") as f:
        true_boxes = pickle.load(f)

    print("Computing mAP...")
    iou_thresholds = torch.tensor([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95], dtype=torch.float32)
    mAP_values = mean_average_precision(
        pred_boxes,
        true_boxes,
        iou_thresholds,
        NUM_CLASSES,
    )

    for c in range(NUM_CLASSES):
        mAP_per_class = mAP_values[c, :].mean().item() * 100.0
        print("="*50)
        print(f"[CLASS {c+1}] mAP@0.50:0.95 {round(mAP_per_class, 3)} %")
        print("="*50)
        for i, iou in enumerate(iou_thresholds):
            mAP_per_class_per_iou = mAP_values[c, i].item() * 100.0
            print(f"\tmAP@{iou:.2f}\t{round(mAP_per_class_per_iou, 3)} %")

    mAP = mAP_values.mean().item() * 100.0
    print("="*50)
    print(f"[OVERALL] mAP@0.5:0.95 {round(mAP, 3)} %")
    print("="*50)
    for i, iou in enumerate(iou_thresholds):
        mAP_per_iou = mAP_values[:, i].mean().item() * 100.0
        print(f"\tmAP@{iou:.2f}\t{round(mAP_per_iou, 3)} %")
            mAP_per_class_per_iou = mAP_values[i, c].item() * 100.0
            print(f"\tmAP@{iou}\t{round(mAP_per_class_per_iou, 3)} %")


if __name__ == "__main__":
    #============DEVICE===============
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Using device {DEVICE}]")

    # ===========CONFIG===============
    config = get_config()
    dataset_cfg = config.dataset
    model_cfg = config.model
    test_cfg = config.test

    # =============TOOLS==============
    checkpoint_handler = CheckpointHandler(config.checkpoint)

    # ===========DATASET==============
    NUM_CLASSES = dataset_cfg.num_classes
    ANCHORS_DICT = get_anchors_dict(dataset_cfg.anchors_file)
    SCALED_ANCHORS = OrderedDict({
        size: anchors * size
        for size, anchors in ANCHORS_DICT.items()
    })

    get_bboxes_from_test_set()
    test_metrics()
