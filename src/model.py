import warnings
from collections import OrderedDict
from functools import partial

import torch
import torchvision
from munch import Munch
from torch import Tensor
from transformers import BeitModel


class DocumentObjectDetector(torch.nn.Module):
    def __init__(self, config: Munch):
        super().__init__()

        self.C: int = int(config.num_classes)
        self.FPN_CHANNELS: int = int(config.fpn_channels)
        self.BACKBONE_CHECKPOINT: str = config.backbone_checkpoint

        if self.BACKBONE_CHECKPOINT is None or self.BACKBONE_CHECKPOINT == "":
            warnings.warn("backbone_checkpoint is not set. Using default checkpoint.")
            self.BACKBONE_CHECKPOINT = "microsoft/dit-large"

        # DiT backbone
        self.backbone = BeitModel.from_pretrained(self.BACKBONE_CHECKPOINT, add_pooling_layer=False)
        self.backbone_forward = partial(self.backbone.forward, output_attentions=True, output_hidden_states=True, return_dict=True)

        # Feature map rescalers
        # conv_output_size = (input_size - kernel_size + 2 * padding) / stride + 1
        # conv_transpose_output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
        self.rescalers = torch.nn.ModuleList([
            # 4x upscaling
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2),
                torch.nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2),
            ),
            # 2x upscaling
            torch.nn.ConvTranspose2d(in_channels=1024, out_channels=1024, kernel_size=2, stride=2),
            # identity
            torch.nn.Identity(),
            # 2x downscaling
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        self.num_patches_per_side: int = self.backbone.config.image_size // self.backbone.config.patch_size
        self.num_patches: int = self.num_patches_per_side ** 2
        self.num_hidden_layers: int = self.backbone.config.num_hidden_layers
        self.hidden_size: int = self.backbone.config.hidden_size

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
            self.backbone.config.hidden_size, # patch embedding size
            self.num_patches_per_side, # number of patches along the horizontal axis #? W or H?
            self.num_patches_per_side # number of patches along the vertical axis #? W or H?
        )
        self.fpn = torchvision.ops.FeaturePyramidNetwork([self.hidden_size]*len(self.fpn_layers), self.FPN_CHANNELS)

        # YOLOv3 head (1x1 convolutions)
        self.yolo_head = torch.nn.Conv2d(in_channels=self.FPN_CHANNELS, out_channels=3 * (5 + self.C), kernel_size=1)

    def forward(self, pixel_values: Tensor):
        # Backbone forward pass
        backbone_outputs = self.backbone_forward(pixel_values)
        # Skip the first hidden state, which is just the embedded input
        hidden_states: tuple[Tensor] = backbone_outputs.hidden_states[1:]

        # Extract feature maps from hidden states
        feature_maps: Tensor = torch.stack([hidden_states[i] for i in self.fpn_layers], dim=1) # (B, 4, 197, 1024)
        feature_maps = feature_maps[:, :, 1:, :] # remove cls token (B, 4, 196, 1024)
        feature_maps = feature_maps.transpose(2, 3) # (B, 4, 1024, 196)
        feature_maps = feature_maps.reshape(self.FEAT_MAPS_SHAPE) # (B, 4, 1024, 14, 14)
        fpn_input: dict[str, Tensor] = OrderedDict(
            {
                self.fpn_layer_names[i]: self.rescalers[i](feature_maps[:, i, ...])
                for i in range(len(self.fpn_layers))
            })

        # FPN forward pass
        fpn_output: dict[str, Tensor] = self.fpn(fpn_input)

        # YOLOv3 head forward pass
        yolo_output: dict[int, Tensor] = OrderedDict()
        for feat_map in fpn_output.values():
            feat_map_size: int = feat_map.shape[-1]

            output = self.yolo_head(feat_map) # (B, 3 * (5 + self.C), feat_map_size, feat_map_size)
            output = output.reshape(-1, 3, 5 + self.C, feat_map_size, feat_map_size) # (B, 3, 5 + self.C, feat_map_size, feat_map_size)
            output = output.transpose(2, 4) # (B, 3, feat_map_size, feat_map_size, 5 + self.C)
            yolo_output.update({feat_map_size: output})

        return yolo_output


def test_model():
    B = 8 # batch size
    C = 11 # number of classes

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = DocumentObjectDetector(Munch(num_classes=C, fpn_channels=512))
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
