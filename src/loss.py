import torch
import torch.nn as nn
from torch import Tensor

from metrics.iou import intersection_over_union


OBJ = list(range(0, 1))
BBOX = list(range(1, 5))
CLS = list(range(5, 16))

BBOX_POS = list(range(1, 3))
BBOX_SIZE = list(range(3, 5))


class YoloLossPerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, predictions: Tensor, target: Tensor, anchors: Tensor) -> Tensor:
        """
        Compute the YOLOv3 loss for a single scale

        B=Batch size, S=Grid size, A=Number of bounding boxes, C=Number of classes

        :param predictions: (B, A, S, S, 5+C) tensor containing the predictions
        :param target: (B, A, S, S, 5+C) tensor containing the targets
        :param anchor_sizes: (A, 2) tensor where 2 is width + height (normalized by the feature map size)

        :return: (1) tensor containing the YOLOv3 loss for the given predictions and targets
        """

        # B = predictions.shape[0]
        # A = predictions.shape[1]
        S = predictions.shape[2]
        C = predictions.shape[-1] - 5

        # Get the obj and noobj masks
        obj_mask = target[..., OBJ] == 1
        noobj_mask = target[..., OBJ] == 0
        # ignored_mask = target[..., OBJ] == -1

        # ==================== #
        #    NO_OBJECT LOSS    #
        # ==================== #
        predicted_no_objectness = torch.masked_select(predictions[..., OBJ], noobj_mask)

        no_object_loss = self.bce(predicted_no_objectness, torch.zeros_like(predicted_no_objectness))

        # Check if there are any objects in the target
        NUM_OBJ = obj_mask.sum()
        if NUM_OBJ == 0:
            return torch.tensor([0.0, 0.0, no_object_loss, 0.0], dtype=torch.float32)

        # ==================== #
        #          IoU         #
        # ==================== #
        # Predicted box coordinates
        predicted_boxes = torch.cat([
            self.sigmoid(predictions[..., BBOX_POS].detach().clone()),
            torch.exp(predictions[..., BBOX_SIZE].detach().clone()) * anchors.reshape(1, 3, 1, 1, 2)
        ], dim=-1)
        predicted_boxes = torch.masked_select(predicted_boxes, obj_mask).reshape(-1, 4)

        # Target box coordinates
        target_boxes = torch.masked_select(target[..., BBOX], obj_mask).reshape(-1, 4)

        ious = intersection_over_union(predicted_boxes, target_boxes)


        # ==================== #
        #      OBJECT LOSS     #
        # ==================== #
        predicted_objectness = torch.masked_select(predictions[..., OBJ], obj_mask)

        object_loss = self.bce(predicted_objectness, ious)

        # ==================== #
        #       BOX LOSS       #
        # ==================== #
        predicted_boxes = torch.masked_select(predictions[..., BBOX], obj_mask).reshape(-1, 4)
        predicted_boxes[:, [0, 1]] = self.sigmoid(predicted_boxes[:, [0, 1]])

        target_boxes = target[..., BBOX].clone()
        target_boxes[..., [2, 3]] = torch.log(1e-16 + target_boxes[..., [2, 3]] / anchors.reshape(1, 3, 1, 1, 2))
        target_boxes = torch.masked_select(target_boxes, obj_mask).reshape(-1, 4)

        box_loss = self.mse(predicted_boxes, target_boxes)

        # ==================== #
        #      CLASS LOSS      #
        # ==================== #
        predicted_classes = torch.masked_select(predictions[..., CLS], obj_mask).reshape(-1, C)
        target_classes = torch.masked_select(target[..., CLS], obj_mask).reshape(-1, C)

        class_loss = self.bce(predicted_classes, target_classes)

        # ==================== #
        #       YOLO LOSS      #
        # ==================== #
        no_object_loss *= 0.5 # λnoobj
        box_loss *= 5.0 # λcoord


        yolo_loss = torch.stack([box_loss, object_loss, no_object_loss, class_loss], dim=0)
        return yolo_loss


class YoloLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.yolo_loss_per_scale = YoloLossPerScale()

    def forward(self, predictions: dict[int, Tensor], target: dict[int, Tensor], anchors: dict[int, Tensor]) -> Tensor:
        scale_sizes: tuple[int] = tuple(predictions.keys())
        yolo_losses: Tensor = torch.stack(
            [
                self.yolo_loss_per_scale(predictions[size], target[size], anchors[size])
                for size in scale_sizes
            ], dim=0)
        return yolo_losses.mean(axis=0).sum()
