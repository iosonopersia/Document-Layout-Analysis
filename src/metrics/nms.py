import torch
from metrics.iou import intersection_over_union

def non_max_suppression(bboxes: torch.Tensor, iou_threshold: float, confidence_threshold: float, num_classes: int=11) -> torch.Tensor:
    """
    Performs non-maximum suppression (NMS) on inference results

        :param bboxes: tensor of shape (N, 6) containing predicted bounding boxes for a single image
        :param iou_threshold: threshold for IoU score to detect overlapping bounding boxes
        :param threshold: threshold for objectness score to detect bounding boxes
        :param num_classes: number of classes in the dataset
        :return: tensor of shape (N, 6) containing predicted bounding boxes for a single image after NMS
    """
    bboxes_per_class = []
    for c in range(num_classes):
        bboxes_per_class.append(nms_per_class(bboxes, iou_threshold, confidence_threshold, c))

    bboxes_after_nms = torch.cat(bboxes_per_class, dim=0)
    return bboxes_after_nms


def nms_per_class(bboxes: torch.Tensor, iou_threshold: float, threshold: float, class_id: int) -> torch.Tensor:
    CLASS_ID = [1]
    OBJ_SCORE = [2]

    # Bounding boxes are filtered by class and (minimum) objectness score
    class_mask = bboxes[:, CLASS_ID] == class_id
    objectness_mask = bboxes[:, OBJ_SCORE] > threshold
    selection_mask = torch.logical_and(class_mask, objectness_mask)
    selection_mask = selection_mask.repeat(1, 7)

    bboxes = bboxes[selection_mask].reshape(-1, 7)

    # Bounding boxes are sorted by objectness score
    sorted_indices = bboxes[:, OBJ_SCORE].flatten().argsort(stable=False, dim=0, descending=True)
    bboxes = bboxes[sorted_indices, :]

    bbox_ids: list[int] = list(range(bboxes.shape[0]))
    bbox_ids_after_nms: list[int] = []

    while len(bbox_ids) > 0:
        i = bbox_ids.pop(0)
        chosen_box = bboxes[i, :]

        ious = intersection_over_union(chosen_box, bboxes[bbox_ids, :])
        ids_to_remove = set(torch.argwhere(ious > iou_threshold).flatten().tolist())

        # Remove all bounding boxes that overlap with the chosen bounding box
        bbox_ids = [bbox_ids[j] for j in range(len(bbox_ids)) if j not in ids_to_remove]

        bbox_ids_after_nms.append(i)

    return bboxes[bbox_ids_after_nms, :]
