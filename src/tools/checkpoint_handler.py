import os
from typing import Any, Optional

import torch
from munch import Munch


class CheckpointHandler():
    def __init__(self, config: Munch) -> None:
        if config is None:
            raise ValueError("CheckpointHandler: config cannot be None")

        self.config = config
        self.save_checkpoint: bool = config.save_checkpoint if hasattr(config, 'save_checkpoint') else False
        self.load_checkpoint: bool = config.load_checkpoint if hasattr(config, 'load_checkpoint') else False

        # Save checkpoint
        if self.save_checkpoint:
            if not hasattr(config, 'save_path'):
                raise ValueError("CheckpointHandler: save_path cannot be None if save_checkpoint is True")
            self.save_path: str = os.path.abspath(config.save_path)
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True) # Create folders if they don't exist

        # Load checkpoint
        if self.load_checkpoint:
            if not hasattr(config, 'load_path'):
                raise ValueError("CheckpointHandler: load_path cannot be None if load_checkpoint is True")
            self.load_path: str = os.path.abspath(config.load_path)
            if not os.path.isfile(self.load_path):
                raise FileNotFoundError(f"CheckpointHandler: Checkpoint file {self.load_path} does not exist")

    def save(self, epoch: int, model_state_dict: dict, optimizer_state_dict: dict, save_path: Optional[str] = None) -> None:
        save_path = save_path if save_path is not None else self.save_path
        if self.save_checkpoint:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
            }, save_path)

    def restore(self, model: Any, optimizer: Any) -> tuple[int, bool]:
        start_epoch: int = 0
        checkpoint_restored: bool = False
        if self.load_checkpoint:
            print(f"CheckpointHandler: restoring checkpoint from {self.load_path}")
            checkpoint = torch.load(self.load_path)
            start_epoch = checkpoint['epoch'] + 1 # start from the next epoch
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            checkpoint_restored = True

        return start_epoch, checkpoint_restored
