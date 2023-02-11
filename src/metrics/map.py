from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from metrics.iou import intersection_over_union


def mean_average_precision(pred_boxes: torch.Tensor, true_boxes: torch.Tensor, iou_thresholds: torch.Tensor, num_classes: int=11):
    num_iou_thresholds = iou_thresholds.shape[0]
    if num_iou_thresholds < 1:
        raise ValueError(f"Number of IoU thresholds must be greater than 0, got {num_iou_thresholds}")

    mAP_per_class = torch.zeros((num_classes, num_iou_thresholds), dtype=torch.float32)

    loop = tqdm(range(num_classes), leave=True)
    loop.set_description(f"COCOmAP per class")

    for c in loop:
        mAP_per_class[c, :] = average_precision_per_class(pred_boxes, true_boxes, iou_thresholds, c)
        loop.set_postfix(class_id=c+1)

    return mAP_per_class


def average_precision_per_class(pred_boxes: torch.Tensor, true_boxes: torch.Tensor, iou_thresholds: torch.Tensor, class_id: int) -> float:
    SAMPLE_ID = [0]
    CLASS_ID = [1]
    OBJ_SCORE = [2]
    BBOX = [3, 4, 5, 6]

    num_iou_thresholds = iou_thresholds.shape[0]
    if num_iou_thresholds < 1:
        raise ValueError(f"Number of IoU thresholds must be greater than 0, got {num_iou_thresholds}")

    detections_mask = (pred_boxes[:, CLASS_ID] == class_id).repeat(1, 7)
    detections = pred_boxes[detections_mask].reshape(-1, 7)
    ground_truths_mask = (true_boxes[:, CLASS_ID] == class_id).repeat(1, 7)
    ground_truths = true_boxes[ground_truths_mask].reshape(-1, 7)

    num_detections: int = detections.shape[0]
    num_ground_truths: int = ground_truths.shape[0]

    if num_detections == 0 or num_ground_truths == 0:
        # AP is zero if there are no detections or ground truth bboxes for this class
        return 0.0

    img_ids, num_bboxes_per_img = ground_truths[:, SAMPLE_ID].int().unique(sorted=False, return_counts=True, dim=None)
    is_matched_bbox = {
        img_ids[i].item(): torch.zeros((num_bboxes_per_img[i], num_iou_thresholds), dtype=torch.bool)
        for i in range(img_ids.shape[0])
    }
    is_TP = torch.zeros((num_detections, num_iou_thresholds), dtype=torch.bool)

    sorted_indices = detections[:, OBJ_SCORE].flatten().argsort(stable=False, descending=True, dim=0)
    detections = detections[sorted_indices, :]

    for i in range(num_detections):
        detection = detections[i, :]

        detection_img_id = detection[SAMPLE_ID].int().item()
        if detection_img_id not in is_matched_bbox:
            # If this image has no ground truth bboxes, then each detection is a false positive
            continue
        elif is_matched_bbox[detection_img_id].all():
            # If all ground truth bboxes for this image have already been matched to a detection, then this detection is a false positive
            continue

        # IoUs must be calculated for each ground truth bbox from the same image
        sample_id_mask = (ground_truths[:, SAMPLE_ID] == detection_img_id).repeat(1, 7)
        ground_truth_img = ground_truths[sample_id_mask].reshape(-1, 7)

        ious = intersection_over_union(detection[BBOX], ground_truth_img[:, BBOX]) # (num_ground_truths, 1)
        best_gt_idx = torch.argmax(ious, dim=0)
        best_iou = ious[best_gt_idx] # (1,)

        is_TP_mask = best_iou > iou_thresholds # (num_iou_thresholds,)
        is_not_matched = torch.logical_not(is_matched_bbox[detection_img_id][best_gt_idx, :]) # (num_iou_thresholds,)

        is_TP[i, :] = torch.logical_and(is_TP_mask, is_not_matched)
        is_matched_bbox[detection_img_id][best_gt_idx, :] = torch.logical_or(is_TP_mask, is_matched_bbox[detection_img_id][best_gt_idx, :])

    # Compute precision and recall at each detection threshold
    TP_cumsum = torch.cumsum(is_TP, dim=0)
    precisions = torch.div(TP_cumsum, torch.arange(1, num_detections + 1, dtype=torch.float32).unsqueeze(dim=-1))
    recalls = TP_cumsum / num_ground_truths

    ZERO = torch.zeros((1, num_iou_thresholds), dtype=torch.float32)
    ONE = torch.ones((1, num_iou_thresholds), dtype=torch.float32)
    # First point is always (0, 1)
    recalls = torch.cat([ZERO, recalls], dim=0)
    precisions = torch.cat([ONE, precisions], dim=0)

    # Save the plot figure of the precision-recall curve
    save_precision_recall_plot(precisions, recalls, iou_thresholds, class_id)

    # Numerical integration of the precision-recall curve
    average_precision = torch.trapezoid(y=precisions, x=recalls, dim=0)
    return average_precision


def save_precision_recall_plot(precisions: torch.Tensor, recalls: torch.Tensor, iou_thresholds: torch.Tensor, class_id: int) -> None:
    fig = plt.figure()
    plt.plot(recalls, precisions)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curves for class {class_id + 1}")
    plt.legend([f"IoU {iou:.2f}" for iou in iou_thresholds])
    plt.savefig(f"./data/mAP_plots/mAP-class-{class_id + 1}.png")
    plt.close(fig)
