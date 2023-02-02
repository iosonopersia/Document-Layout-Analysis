import torch
from tqdm import tqdm

from metrics.nms import non_max_suppression


def get_evaluation_bboxes(
    loader,
    model,
    iou_threshold,
    anchors,
    confidence_threshold,
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    for batch in tqdm(loader):
        images = batch['image'].to(device) # (B, 3, 224, 224)
        targets = batch['target']
        image_ids = batch['image_id'] # (B,)
        B = image_ids.numel()

        with torch.inference_mode():
            predictions = model(images)

        for S in predictions.keys():
            predictions[S] = predictions[S].cpu()

        # ===========================
        # Get bboxes from predictions
        # ===========================
        bboxes = predictions_to_bboxes(predictions, image_ids, anchors) # (#PRED_BOXES, 7)

        nms_boxes_per_image = []
        for i in range(B):
            # Apply NMS to each image separately
            nms_boxes = non_max_suppression(
                bboxes[i, ...],
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold
            )
            nms_boxes_per_image.append(nms_boxes)
        nms_boxes = torch.cat(nms_boxes_per_image, dim=0) # (#NMS_BOXES, 7)

        # =======================
        # Get ground truth bboxes
        # =======================
        true_boxes = targets_to_bboxes(targets, image_ids) # (#GT_BOXES, 7)

        all_pred_boxes.append(nms_boxes)
        all_true_boxes.append(true_boxes)

    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_true_boxes = torch.cat(all_true_boxes, dim=0)

    return all_pred_boxes, all_true_boxes


def predictions_to_bboxes(predictions: dict[int, torch.Tensor], image_ids: torch.Tensor, anchors: dict[int, torch.Tensor]) -> torch.Tensor:
    bboxes = []
    for S in predictions.keys():
        scaled_anchors = anchors[S] * S
        B = predictions[S].shape[0]

        pred_box = predictions[S][..., 1:5] # (B, 3, S, S, 4)
        pred_score = torch.sigmoid(predictions[S][..., 0:1]) # (B, 3, S, S, 1)
        pred_class = torch.argmax(predictions[S][..., 5:], dim=-1, keepdim=True) # (B, 3, S, S, 1)
        pred_image_id = image_ids.reshape(B, 1, 1, 1, 1).repeat(1, 3, S, S, 1) # (B, 3, S, S, 1)

        # ===================================================
        # Apply required transformations to model predictions
        # ===================================================
        scaled_anchors = scaled_anchors.reshape(1, 3, 1, 1, 2)
        pred_box[..., 0:2] = torch.sigmoid(pred_box[..., 0:2])
        pred_box[..., 2:4] = torch.exp(pred_box[..., 2:4]) * scaled_anchors

        # =============================================
        # Convert box positions from (dx, dy) to (x, y)
        # =============================================
        cell_indices = torch.arange(S, dtype=torch.float32)
        cell_i = cell_indices.reshape(1, 1, S, 1, 1).repeat(B, 3, 1, S, 1)
        cell_j = cell_indices.reshape(1, 1, 1, S, 1).repeat(B, 3, S, 1, 1)
        cell_indices = torch.cat((cell_i, cell_j), dim=-1)

        pred_box[..., 0:2] += cell_indices

        # ================================================
        # Scale box coordinates to the original image size
        # ================================================
        pred_box *= (1025 / S)

        predicted_bboxes = torch.cat([pred_image_id, pred_class, pred_score, pred_box], dim=-1).reshape(B, 3*S*S, 7)
        bboxes.append(predicted_bboxes)

    bboxes = torch.cat(bboxes, dim=1) # (B, T, 7), T = 3*(7^2 + 14^2 + 28^2 + 56^2)
    return bboxes


def targets_to_bboxes(targets: dict[int, torch.Tensor], image_ids: torch.Tensor) -> torch.Tensor:
    OBJ_SCORE = 0
    # We can consider just one scale, since bboxes are the same across all scales
    # We choose the last scale, which is the smallest one, to speed up the process
    S = list(targets.keys())[-1]
    B = targets[S].shape[0]

    true_boxes_per_image = []
    for i in range(B):
        tgt_per_image = targets[S][[i], ...]

        # ================================================
        # Filter out non-object bounding boxes from target
        # ================================================
        obj_mask = tgt_per_image[..., [OBJ_SCORE]] == 1
        tgt = torch.masked_select(tgt_per_image, obj_mask).reshape(-1, 5+11) # (N, 5+11)
        N = tgt.shape[0]
        tgt_class = torch.argmax(tgt[..., 5:16], dim=-1, keepdim=True) # (N, 1)
        tgt_image_id = image_ids[[i]].repeat(N, 1) # (N, 1)

        # =============================================
        # Convert box positions from (dx, dy) to (x, y)
        # =============================================
        cell_indices = torch.argwhere(obj_mask)[:, [3, 2]] # (j, i) for each bbox (N, 2)
        tgt[:, [1, 2]] += cell_indices

        # ================================================
        # Scale box coordinates to the original image size
        # ================================================
        tgt[:, 1:5] *= (1025 / S)

        tgt = torch.cat([tgt_image_id, tgt_class, tgt[..., 0:5]], dim=-1) # (N, 1) + (N, 1) + (N, 5) -> (N, 7)
        true_boxes_per_image.append(tgt)

    true_boxes = torch.cat(true_boxes_per_image, dim=0) # (#GT_BOXES, 7)
    return true_boxes

