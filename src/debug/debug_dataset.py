import warnings

import torch
from torch.utils.data import Dataset
from transformers import BeitFeatureExtractor, logging

from utils import get_anchors_dict, get_config

logging.set_verbosity_error()


class COCODataset(Dataset):
    def __init__(self, anchors_dict = {}, feature_extractor=None):
        if anchors_dict is None or len(anchors_dict) == 0:
            raise ValueError("anchors must be a non-empty dictionary.")

        if feature_extractor is None:
            raise ValueError("feature_extractor must be provided.")

        self.feature_extractor = feature_extractor

        # Grid sizes
        self.S: list[int] = list(anchors_dict.keys()) # Grid sizes
        self.num_scales: int = len(self.S)

        # Anchors
        self.anchor_shapes: dict[int, list[float]] = anchors_dict
        self.num_anchors_per_scale: int = len(anchors_dict[self.S[0]]) # Here we assume same num of anchors per scale

    def __len__(self):
        return 8192

    def __getitem__(self, index):
        image = torch.full((3, 1025, 1025), 255, dtype=torch.int16)

        obj_class = torch.randint(1, 12, (1,)) # Randomly select an object class
        obj_color = (obj_class - 1) * (255 / 11) # [0, 10] -> [0, 232] (don't use 255 because is the background color)
        image[:, 412:613, 412:613] = int(obj_color) # Set object color

        # from matplotlib import pyplot as plt
        # plt.imshow(image.permute(1, 2, 0).numpy())

        # Create targets
        targets = {S: torch.zeros((self.num_anchors_per_scale, S, S, 5+11), dtype=torch.float32) for S in self.S}
        for S in self.S:
            if S % 2 != 0:
                delta_x, delta_y = 0.5, 0.5
            else:
                delta_x, delta_y = 0.0, 0.0

            obj_w, obj_h = 0.195121951, 0.195121951 # 200px / 1025px

            targets[S][2, S//2, S//2, 0] = 1.0 # Set objectness score to 1
            targets[S][2, S//2, S//2, 1:5] = torch.tensor([delta_x, delta_y, S*obj_w, S*obj_h], dtype=torch.float32) # Set bbox coords
            targets[S][2, S//2, S//2, 5+(obj_class-1)] = 1.0 # Set object class

         # Extract features
        image = self.feature_extractor(image, return_tensors='pt').pixel_values # (1, C, H, W)
        image = image.squeeze(dim=0) # (C, H, W)

        return {'image_id': index, 'image': image, 'target': targets}


feature_extractor: BeitFeatureExtractor = None

def get_dataloader(split: str = "train") -> tuple[COCODataset, torch.utils.data.DataLoader]:
    config = get_config()

    # Load feature extractor only once
    global feature_extractor
    if feature_extractor is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_cfg = config.model
            feature_extractor = BeitFeatureExtractor.from_pretrained(model_cfg.backbone)

    # Select config for the required split
    if split in ["train", "val", "test"]:
        config_dataloader = config.dataloader[split]
    else:
        raise ValueError("Invalid split")

    # Create dataset and dataloader
    dataset_cfg = config.dataset
    dataset = COCODataset(
        anchors_dict=get_anchors_dict(dataset_cfg.anchors_file),
        feature_extractor=feature_extractor
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        **config_dataloader
    )

    return dataloader
