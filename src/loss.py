import torch
import torch.nn as nn
from torch import Tensor

from yolo_utils import intersection_over_union

OBJ = list(range(0, 1))
BBOX = list(range(1, 5))
CLS_PRED = list(range(5, 16))
CLS_TGT = list(range(5, 6))


class YoloLossPerScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Multiplicative weights for each part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions: Tensor, target: Tensor, anchor_sizes: Tensor) -> Tensor:
        """
        B=Batch size, S=Grid size, A=Number of bounding boxes, C=Number of classes
            * `predictions` is a tensor of size (B, A, S, S, 5+C)
            * `target` is a tensor of size (B, A, S, S, 5+1) where 1 is the class index
            * `anchors` is a tensor of size (A, 2) where 2 is width + height (normalized by the feature map size)
        """
        # Get the obj and noobj masks (when target == -1, the prediction is ignored for the computation of the loss)
        obj_mask = target[..., OBJ] == 1  # in paper this is Iobj_i
        noobj_mask = target[..., OBJ] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        # Objectness score should be 0 if there is no object
        no_object_loss = self.bce(
            torch.masked_select(predictions[..., OBJ], noobj_mask),
            torch.masked_select(target[..., OBJ], noobj_mask)
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        # Objectness score should be IoU(pred, tgt) if there is an object

        obj_tensor_indices = torch.argwhere(obj_mask) # (num_objects, 4)
        obj_anchors_indices = obj_tensor_indices[:, 1] # (num_objects) anchor index for each object
        obj_anchors = anchor_sizes[obj_anchors_indices] # (num_objects, 2) anchor size for each object (width, height)
        obj_grid_indices = obj_tensor_indices[:, [3, 2]] # (num_objects, 2) grid cell index for each object (j, i), meaning (x, y)
        num_objects = obj_tensor_indices.shape[0]

        predicted_boxes = torch.masked_select(predictions[..., BBOX], obj_mask)
        predicted_boxes = predicted_boxes.reshape(num_objects, 4)

        predicted_positions = obj_grid_indices + self.sigmoid(predicted_boxes[:, [0, 1]]) # (x, y)
        predicted_sizes = obj_anchors * torch.exp(predicted_boxes[:, [2, 3]]) # (width, height)

        target_boxes = torch.masked_select(target[..., BBOX], obj_mask)
        target_boxes = target_boxes.reshape(num_objects, 4)

        ious = intersection_over_union(
            torch.cat([predicted_positions, predicted_sizes], dim=-1).detach(), # detach from the computation graph
            target_boxes,
            box_format='midpoint')

        object_loss = self.mse(
            torch.masked_select(self.sigmoid(predictions[..., OBJ]), obj_mask),
            ious
        )

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # The box coordinates should be close to the target box coordinates
        # The box dimensions should be close to the target box dimensions
        predicted_boxes[:, [0, 1]] = self.sigmoid(predicted_boxes[:, [0, 1]]) # x, y adjustments

        target_boxes[:, [0, 1]] = target_boxes[:, [0, 1]] - obj_grid_indices # x, y adjustments
        target_boxes[:, [2, 3]] = torch.log(1e-16 + target_boxes[:, [2, 3]] / obj_anchors) # width, height adjustments

        box_loss = self.mse(
            predicted_boxes,
            target_boxes
        )

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        # The class prediction should be close to the target class
        class_loss = self.entropy(
            torch.masked_select(predictions[..., CLS_PRED], obj_mask).reshape(-1, len(CLS_PRED)),
            torch.masked_select(target[..., CLS_TGT], obj_mask).long())

        # ================== #
        #     TOTAL LOSS     #
        # ================== #
        box_loss *= self.lambda_box
        object_loss *= self.lambda_obj
        no_object_loss *= self.lambda_noobj
        class_loss *= self.lambda_class

        yolo_loss = torch.stack([box_loss, object_loss, no_object_loss, class_loss], dim=0)
        return yolo_loss.sum()


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
        return yolo_losses.mean()
