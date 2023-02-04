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
    OBJ_SCORE = [0]
    BOX_COORDS = [1, 2, 3, 4]
    BOX_POS = [0, 1]
    BOX_SIZE = [2, 3]

    bboxes = []
    for S in predictions.keys():
        B = predictions[S].shape[0]
        C = predictions[S].shape[-1] - 5
        CLASSES = list(range(5, 5 + C))

        pred_score = torch.sigmoid(predictions[S][..., OBJ_SCORE]) # (B, 3, S, S, 1)
        pred_class = torch.argmax(predictions[S][..., CLASSES], dim=-1, keepdim=True) # (B, 3, S, S, 1)
        pred_image_id = image_ids.reshape(B, 1, 1, 1, 1).repeat(1, 3, S, S, 1) # (B, 3, S, S, 1)

        # ===================================================
        # Apply required transformations to model predictions
        # ===================================================
        pred_box = predictions[S][..., BOX_COORDS] # (B, 3, S, S, 4)
        pred_box[..., BOX_POS] = torch.sigmoid(pred_box[..., BOX_POS])
        pred_box[..., BOX_SIZE] = torch.exp(pred_box[..., BOX_SIZE]) * anchors[S].reshape(1, 3, 1, 1, 2)

        # =============================================
        # Convert box positions from (dx, dy) to (x, y)
        # =============================================
        cell_indices = torch.arange(S, dtype=torch.int16)
        cell_i = cell_indices.reshape(1, 1, S, 1, 1).repeat(B, 3, 1, S, 1)
        cell_j = cell_indices.reshape(1, 1, 1, S, 1).repeat(B, 3, S, 1, 1)
        cell_indices = torch.cat((cell_j, cell_i), dim=-1)

        pred_box[..., BOX_POS] += cell_indices.float()

        # ================================================
        # Scale box coordinates to the original image size
        # ================================================
        pred_box *= (1025 / S)

        predicted_bboxes = torch.cat([pred_image_id, pred_class, pred_score, pred_box], dim=-1).reshape(B, 3*S*S, 7)
        bboxes.append(predicted_bboxes)

    bboxes = torch.cat(bboxes, dim=1) # (B, T, 7), T = 3*(7^2 + 14^2 + 28^2 + 56^2)
    return bboxes


def targets_to_bboxes(targets: dict[int, torch.Tensor], image_ids: torch.Tensor) -> torch.Tensor:
    # We can consider just one scale, since bboxes are the same across all scales
    # We choose the last scale, which is the smallest one, to speed up the process
    S = list(targets.keys())[-1]
    B = targets[S].shape[0]
    C = targets[S].shape[-1] - 5

    OBJ_SCORE = [0]
    BOX_COORDS = [1, 2, 3, 4]
    BOX_POS = [0, 1]
    # BOX_SIZE = [2, 3]
    CLASSES = list(range(5, 5 + C))

    true_boxes_per_image = []
    for i in range(B):
        tgt_per_image = targets[S][i, ...] # (3, S, S, 16)

        # ================================================
        # Filter out non-object bounding boxes from target
        # ================================================
        obj_mask = tgt_per_image[..., OBJ_SCORE] == 1 # (3, S, S, 1)
        tgt = tgt_per_image[obj_mask.repeat(1, 1, 1, 5 + C)].reshape(-1, 5 + C) # (N, 5+C)
        N = tgt.shape[0]

        tgt_image_id = image_ids[[i]].repeat(N, 1) # (N, 1)
        tgt_class = torch.argmax(tgt[:, CLASSES], dim=-1, keepdim=True) # (N, 1)
        tgt_score = torch.ones((N, 1), dtype=torch.float32) # (N, 1)
        tgt_boxes = tgt[:, BOX_COORDS] # (N, 4)

        # =============================================
        # Convert box positions from (dx, dy) to (x, y)
        # =============================================
        cell_indices = torch.argwhere(obj_mask)[:, [2, 1]] # (j, i) for each bbox (N, 2)
        tgt_boxes[:, BOX_POS] += cell_indices

        # ================================================
        # Scale box coordinates to the original image size
        # ================================================
        tgt_boxes *= (1025 / S)

        tgt = torch.cat([tgt_image_id, tgt_class, tgt_score, tgt_boxes], dim=-1) # (N, 1) + (N, 1) + (N, 1) + (N, 4) -> (N, 7)
        true_boxes_per_image.append(tgt)

    true_boxes = torch.cat(true_boxes_per_image, dim=0) # (#GT_BOXES, 7)
    return true_boxes

