import csv
import os

import torch
from munch import Munch

config: Munch = None

def get_config() -> Munch:
    global config
    if config is None:
        if not os.path.exists('./src/config.yaml'):
            raise FileNotFoundError('Config file not found')

        with open('./src/config.yaml', 'rt', encoding='utf-8') as f:
            config = Munch.fromYAML(f.read())
    return config


def get_anchors_dict(filepath: str) -> dict[int, list[float]]:
    filepath = os.path.abspath(filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError("Anchors file not found")

    with open(filepath, "rt") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader) # Skip header
        anchors = [[float(x) for x in row] for row in reader]

    if len(anchors) != 12:
        raise ValueError("Anchors file should contain 12 rows")

    # Sort anchors by area
    anchors = sorted(anchors, key=lambda x: x[0] * x[1]) # ascending order

    anchors_dict = {
        56: torch.tensor(anchors[0:3], dtype=torch.float32),
        28: torch.tensor(anchors[3:6], dtype=torch.float32),
        14: torch.tensor(anchors[6:9], dtype=torch.float32),
        7: torch.tensor(anchors[9:12], dtype=torch.float32)
    }

    return anchors_dict

