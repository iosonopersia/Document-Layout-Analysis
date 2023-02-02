import torch
from tqdm import tqdm
from metrics.iou import intersection_over_union


def mean_average_precision(pred_boxes: torch.Tensor, true_boxes: torch.Tensor, num_classes: int=11):
    iou_thresholds = torch.arange(0.5, 1.0, 0.05, dtype=torch.float32)
    mAP_per_iou = torch.zeros_like(iou_thresholds)

    for i in tqdm(range(iou_thresholds.numel())):
        iou_threshold = iou_thresholds[i].item()

        AP_per_class = torch.zeros(num_classes, dtype=torch.float32)
        for c in range(num_classes):
            AP_per_class[c] = average_precision_per_class(pred_boxes, true_boxes, iou_threshold, c)

        mAP_per_iou[i] = AP_per_class.mean()

    return mAP_per_iou.mean()


def average_precision_per_class(pred_boxes: torch.Tensor, true_boxes: torch.Tensor, iou_threshold: float, class_id: int) -> float:
    SAMPLE_ID = 0
    CLASS_ID = 1
    OBJ_SCORE = 2
    BBOX = list(range(3, 7))

    detections = torch.masked_select(pred_boxes, pred_boxes[:, [CLASS_ID]] == class_id).reshape(-1, 7)
    ground_truths = torch.masked_select(true_boxes, true_boxes[:, [CLASS_ID]] == class_id).reshape(-1, 7)

    num_detections: int = detections.shape[0]
    num_ground_truths: int = ground_truths.shape[0]

    if num_detections == 0 or num_ground_truths == 0:
        # AP is zero if there are no detections or ground truth bboxes for this class
        return 0.0

    img_ids, num_bboxes_per_img = ground_truths[:, SAMPLE_ID].int().unique(sorted=False, return_counts=True, dim=None)
    is_matched_bbox = {img_ids[i].item(): torch.zeros(num_bboxes_per_img[i], dtype=torch.bool) for i in range(img_ids.shape[0])}

    sorted_indices = detections[:, OBJ_SCORE].argsort(stable=False, dim=0, descending=True)
    detections = detections[sorted_indices]
    is_TP = torch.zeros(num_detections, dtype=torch.bool)

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
        ground_truth_img = torch.masked_select(ground_truths, ground_truths[:, [SAMPLE_ID]] == detection_img_id).reshape(-1, 7)

        ious = intersection_over_union(detection[BBOX], ground_truth_img[:, BBOX])
        best_gt_idx = torch.argmax(ious)
        best_iou = ious[best_gt_idx]

        if best_iou > iou_threshold and not is_matched_bbox[detection_img_id][best_gt_idx]:
            is_TP[i] = True
            is_matched_bbox[detection_img_id][best_gt_idx] = True

    # Compute precision and recall at each detection threshold
    if is_TP.sum() == 0:
        # AP is zero if there are no true positives
        return 0.0

    TP_cumsum = torch.cumsum(is_TP, dim=0)
    precisions = TP_cumsum / torch.arange(1, num_detections + 1, dtype=torch.float32)
    recalls = TP_cumsum / (num_ground_truths + 1e-6)

    # Add the initial point of the ROC curve (prec=1, recall=0)
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))

    # Numerical integration of the precision-recall curve
    average_precision = torch.trapz(precisions, recalls)
    return average_precision
