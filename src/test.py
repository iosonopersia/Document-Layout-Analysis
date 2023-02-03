from collections import OrderedDict
import torch
from dataset import get_dataloader
from metrics.bboxes_extraction import get_evaluation_bboxes
from metrics.map import mean_average_precision
from model import DocumentObjectDetector
from tools.checkpoint_handler import CheckpointHandler
from utils import get_anchors_dict, get_config

def test_metrics():
    # ========== DATASET=============
    eval_loader = get_dataloader("test")

    # ===========MODEL===============
    model = DocumentObjectDetector(NUM_CLASSES, model_cfg)
    model = model.to(DEVICE)
    checkpoint_handler.load(test_cfg.checkpoint_path, model)

    # ============TEST===============
    print("Extracting bboxes from model predictions on the test set...")
    pred_boxes, true_boxes = get_evaluation_bboxes(
        eval_loader,
        model,
        iou_threshold=test_cfg.nms_iou_threshold,
        anchors=ANCHORS_DICT,
        confidence_threshold=test_cfg.confidence_threshold,
    )
    print("Computing mAP...")
    mAP_values = mean_average_precision(
        pred_boxes,
        true_boxes,
        num_classes=NUM_CLASSES,
    )
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for c in range(NUM_CLASSES):
        mAP_per_class = mAP_values[:, c].mean().item()
        print(f"mAP@0.5:0.95 for class {c+1}: {round(100*mAP_per_class, 3)} %")
        for i, iou in enumerate(iou_thresholds):
            mAP_per_class_per_iou = mAP_values[i, c].item()
            print(f"\tmAP@{iou} for class {c+1}: {round(100*mAP_per_class_per_iou, 3)} %")



if __name__ == "__main__":
    #============DEVICE===============
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Using device {DEVICE}]")

    # ===========CONFIG===============
    config = get_config()
    dataset_cfg = config.dataset
    model_cfg = config.model
    test_cfg = config.test

    # ============TOOLS==============
    checkpoint_handler = CheckpointHandler(config.checkpoint)

    # ===========DATASET==============
    NUM_CLASSES = dataset_cfg.num_classes
    ANCHORS_DICT = get_anchors_dict(dataset_cfg.anchors_file)
    SCALED_ANCHORS = OrderedDict({
        size: anchors * size
        for size, anchors in ANCHORS_DICT.items()
    })

    test_metrics()
