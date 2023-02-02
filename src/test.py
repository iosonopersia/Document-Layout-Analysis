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
    checkpoint_handler.load(model)

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
    mapval = mean_average_precision(
        pred_boxes,
        true_boxes,
        num_classes=NUM_CLASSES,
    )
    print(f"MAP@0.5:0.05:0.95: {mapval.item()}")


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
