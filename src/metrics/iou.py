import torch


def intersection_over_union(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor) -> torch.Tensor:
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (..., 4)
        boxes_labels (tensor): Correct Labels of Boxes (..., 4)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Predicted bounding boxes coordinates
    box1_x1 = boxes_preds[..., 0] - boxes_preds[..., 2] / 2
    box1_y1 = boxes_preds[..., 1] - boxes_preds[..., 3] / 2
    box1_x2 = boxes_preds[..., 0] + boxes_preds[..., 2] / 2
    box1_y2 = boxes_preds[..., 1] + boxes_preds[..., 3] / 2

    # True bounding boxes coordinates
    box2_x1 = boxes_labels[..., 0] - boxes_labels[..., 2] / 2
    box2_y1 = boxes_labels[..., 1] - boxes_labels[..., 3] / 2
    box2_x2 = boxes_labels[..., 0] + boxes_labels[..., 2] / 2
    box2_y2 = boxes_labels[..., 1] + boxes_labels[..., 3] / 2

    # Intersection coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = torch.clamp((x2 - x1), min=0) * torch.clamp((y2 - y1), min=0)
    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
