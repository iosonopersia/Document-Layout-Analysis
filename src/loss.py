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
    def __init__(self, wandb_logger):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.wandb_logger = wandb_logger

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

        TRAIN_MODE = predictions.requires_grad

        # Get the obj and noobj masks
        obj_mask = (target[..., OBJ] == 1).repeat(1, 1, 1, 1, 5 + C)
        noobj_mask = (target[..., OBJ] == 0).repeat(1, 1, 1, 1, 5 + C)
        # ignored_mask = (target[..., OBJ] == -1).repeat(1, 1, 1, 1, 5 + C)

        noobj_predictions = predictions[noobj_mask].reshape(-1, 5+ C)
        obj_predictions = predictions[obj_mask].reshape(-1, 5 + C)

        # noobj_target = target[noobj_mask].reshape(-1, 5 + C)
        obj_target = target[obj_mask].reshape(-1, 5 + C)

        # ==================== #
        #    NO_OBJECT LOSS    #
        # ==================== #
        predicted_no_objectness = noobj_predictions[:, OBJ]

        no_object_loss = self.bce(predicted_no_objectness, torch.zeros_like(predicted_no_objectness))

        # Check if there are any objects in the target
        NUM_OBJ = obj_mask.sum()
        if NUM_OBJ == 0:
            if TRAIN_MODE:
                # Log only during training
                self.wandb_logger.log({
                    f'size_{S}/mean_iou': 1.0,
                    f'size_{S}/box_loss': 0.0,
                    f'size_{S}/object_loss': 0.0,
                    f'size_{S}/no_object_loss': no_object_loss.item(),
                    f'size_{S}/class_loss': 0.0,
                })
            return torch.tensor([0.0, 0.0, no_object_loss, 0.0], dtype=torch.float32)

        # ==================== #
        #          IoU         #
        # ==================== #
        # Predicted box coordinates
        all_box_predictions = torch.cat([
            self.sigmoid(predictions[..., BBOX_POS].detach()),
            torch.exp(predictions[..., BBOX_SIZE].detach()) * anchors.reshape(1, 3, 1, 1, 2)
        ], dim=-1)

        obj_mask_box = obj_mask[..., BBOX]
        predicted_boxes = all_box_predictions[obj_mask_box].reshape(-1, 4)

        # Target box coordinates
        target_boxes = obj_target[:, BBOX]

        ious = intersection_over_union(predicted_boxes, target_boxes)

        # ==================== #
        #      OBJECT LOSS     #
        # ==================== #
        predicted_objectness = obj_predictions[:, OBJ]

        object_loss = self.bce(predicted_objectness, ious)

        # ==================== #
        #       BOX LOSS       #
        # ==================== #
        predicted_boxes = obj_predictions[:, BBOX]
        predicted_boxes[:, [0, 1]] = self.sigmoid(predicted_boxes[:, [0, 1]])

        target_boxes = torch.cat([
            target[..., BBOX_POS],
            target[..., BBOX_SIZE] / anchors.reshape(1, 3, 1, 1, 2)
        ], dim=-1)
        target_boxes[..., [2, 3]] = torch.log(1e-16 + target_boxes[..., [2, 3]])
        target_boxes = target_boxes[obj_mask_box].reshape(-1, 4)

        box_loss = self.mse(predicted_boxes, target_boxes)

        # ==================== #
        #      CLASS LOSS      #
        # ==================== #
        predicted_classes = obj_predictions[:, CLS]
        target_classes = obj_target[:, CLS]

        class_loss = self.bce(predicted_classes, target_classes)

        # ==================== #
        #       YOLO LOSS      #
        # ==================== #
        if TRAIN_MODE:
            # Log only during training
            self.wandb_logger.log({
                f'size_{S}/mean_iou': ious.mean().item(),
                f'size_{S}/box_loss': box_loss.item(),
                f'size_{S}/object_loss': object_loss.item(),
                f'size_{S}/no_object_loss': no_object_loss.item(),
                f'size_{S}/class_loss': class_loss.item(),
            })

        yolo_loss = torch.stack([box_loss, object_loss, no_object_loss, class_loss], dim=0)
        return yolo_loss


class YoloLoss(nn.Module):
    def __init__(self, wandb_logger) -> None:
        super().__init__()
        self.yolo_loss_per_scale = YoloLossPerScale(wandb_logger)

    def forward(self, predictions: dict[int, Tensor], target: dict[int, Tensor], anchors: dict[int, Tensor]) -> Tensor:
        scale_sizes: tuple[int] = tuple(predictions.keys())
        yolo_losses: Tensor = torch.stack(
            [
                self.yolo_loss_per_scale(predictions[size], target[size], anchors[size])
                for size in scale_sizes
            ], dim=0)
        return yolo_losses.mean(axis=0).sum()
