import warnings
from collections import OrderedDict

import torch
import torchvision
from munch import Munch
from torch import Tensor
from transformers import BeitConfig, logging

logging.set_verbosity_error()


class DocumentObjectDetector(torch.nn.Module):
    def __init__(self, num_classes: int, config: Munch):
        super().__init__()

        self.num_classes: int = num_classes
        self.fpn_channels: int = config.fpn_channels
        self.backbone_name: str = config.backbone

        if self.backbone_name is None or self.backbone_name == "":
            warnings.warn("backbone_checkpoint is not set. Using default backbone (microsoft/dit-base).")
            self.backbone_name = "microsoft/dit-base"

        # DiT backbone
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.backbone_config = BeitConfig.from_pretrained(self.backbone_name)


        self.num_patches_per_side: int = self.backbone_config.image_size // self.backbone_config.patch_size
        self.num_patches: int = self.num_patches_per_side ** 2
        self.num_hidden_layers: int = self.backbone_config.num_hidden_layers
        self.hidden_size: int = self.backbone_config.hidden_size

        # Create a CNN with total stride of 16
        self.backbone = torch.nn.Sequential(
            # (3, 224, 224) -> (H, 56, 56)
            torch.nn.Conv2d(in_channels=3, out_channels=self.hidden_size, kernel_size=8, stride=4, padding=2),
            torch.nn.BatchNorm2d(num_features=self.hidden_size),
            torch.nn.GELU(),
            # (H, 56, 56) -> (H, 28, 28)
            torch.nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=self.hidden_size),
            torch.nn.GELU(),
            # (H, 28, 28) -> (H, 14, 14)
            torch.nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(num_features=self.hidden_size),
            torch.nn.GELU(),
        )

        # Feature map rescalers (as implemented by the authors of the DiT paper)
        # See https://github.com/microsoft/unilm/blob/4dfdda9fbe950c73616c65efc5b7f6b1a3d2a60a/dit/object_detection/ditod/deit.py#L262-L275
        self.rescalers = torch.nn.ModuleList([
            # 4x upscaling
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=2,
                    stride=2),
                torch.nn.BatchNorm2d(num_features=self.hidden_size),
                torch.nn.GELU(),
                torch.nn.ConvTranspose2d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=2,
                    stride=2),
                torch.nn.BatchNorm2d(num_features=self.hidden_size),
                torch.nn.GELU(),
            ),
            # 2x upscaling
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=2,
                    stride=2),
                torch.nn.BatchNorm2d(num_features=self.hidden_size),
                torch.nn.GELU(),
            ),
            # identity
            torch.nn.Identity(),
            # 2x downscaling
            torch.nn.Sequential(
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.BatchNorm2d(num_features=self.hidden_size),
                torch.nn.GELU(),
            ),
        ])

        self.fpn_layers: list[int] = [
            self.num_hidden_layers // 3 - 1,
            self.num_hidden_layers // 2 - 1,
            (2 * self.num_hidden_layers) // 3 - 1,
            (3 * self.num_hidden_layers) // 3 - 1]
        self.fpn_layer_names: list[str] = [f'layer_{layer_num}' for layer_num in self.fpn_layers]

        # Feature Pyramid Network
        self.FEAT_MAPS_SHAPE: tuple[int] = (
            -1, # batch size
            len(self.fpn_layers), # number of feature maps
            self.hidden_size, # patch embedding size
            self.num_patches_per_side, # number of patches along the vertical axis
            self.num_patches_per_side # number of patches along the horizontal axis
        )
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[self.hidden_size]*len(self.fpn_layers),
            out_channels=self.fpn_channels,
            norm_layer=torch.nn.BatchNorm2d)

        # YOLOv3 head (1x1 convolutions)
        self.yolo_head = torch.nn.Conv2d(in_channels=self.fpn_channels, out_channels=3 * (5 + self.num_classes), kernel_size=1)

    def forward(self, pixel_values: Tensor):
        # Backbone forward pass
        grid14_14 = self.backbone(pixel_values) # (B, 768, 14, 14)

        fpn_input: dict[str, Tensor] = OrderedDict()
        for i in range(len(self.fpn_layers)):
            feat_map = grid14_14
            layer_name: str = self.fpn_layer_names[i]

            feat_map = self.rescalers[i](feat_map) # apply rescaling
            fpn_input[layer_name] = feat_map

        # FPN forward pass
        fpn_output: dict[str, Tensor] = self.fpn(fpn_input)

        # YOLOv3 head forward pass
        yolo_output: dict[int, Tensor] = OrderedDict()
        for feat_map in fpn_output.values():
            feat_map_size: int = feat_map.shape[-1]

            output = self.yolo_head(feat_map) # (B, 3 * (5 + C), feat_map_size, feat_map_size)
            output = output.reshape(-1, 3, 5 + self.num_classes, feat_map_size, feat_map_size) # (B, 3, 5 + C, feat_map_size, feat_map_size)
            output = output.transpose(2, 4) # (B, 3, feat_map_size, feat_map_size, 5 + C)
            yolo_output.update({feat_map_size: output})

        return yolo_output

    def freeze_backbone(self):
        pass
        return

    def unfreeze_backbone(self):
        pass
        return


def test_model():
    B = 8 # batch size
    C = 11 # number of classes

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = DocumentObjectDetector(11, Munch(fpn_channels=256, backbone='microsoft/dit-base'))
    model.to(device)
    model.eval()

    pixel_values = torch.rand(B, 3, 224, 224, device=device)
    yolo_output = model(pixel_values)

    assert len(yolo_output) == 4
    assert yolo_output[56].shape == (B, 3, 56, 56, 5+C)
    assert yolo_output[28].shape == (B, 3, 28, 28, 5+C)
    assert yolo_output[14].shape == (B, 3, 14, 14, 5+C)
    assert yolo_output[7].shape == (B, 3, 7, 7, 5+C)

    print("Model test passed!")


if __name__ == "__main__":
    test_model()
