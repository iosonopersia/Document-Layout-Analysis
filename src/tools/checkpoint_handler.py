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

    def save(self, epoch: int, val_loss: float, model_state_dict: dict, optimizer_state_dict: dict,
             is_frozen_epoch: bool, save_path: Optional[str] = None) -> None:
        save_path = save_path if save_path is not None else self.save_path
        if self.save_checkpoint:
            torch.save({
                'epoch': epoch,
                'val_loss': val_loss,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': optimizer_state_dict,
                'is_frozen_epoch': is_frozen_epoch
            }, save_path)

    def restore_for_training(self, model: Any, optimizer: Any) -> int:
        start_epoch: int = 0
        if self.load_checkpoint:
            print(f"CheckpointHandler: restoring checkpoint from {self.load_path}")
            checkpoint = torch.load(self.load_path)
            if checkpoint['is_frozen_epoch']:
                raise ValueError("CheckpointHandler: Cannot restore checkpoint from frozen epoch")
            else:
                start_epoch = checkpoint['epoch'] + 1 # start from the next epoch
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return start_epoch

    def load_for_testing(self, checkpoint_path: str, model: Any) -> int:
        checkpoint_path = os.path.abspath(checkpoint_path)
        if os.path.exists(checkpoint_path):
            print(f"CheckpointHandler: loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError(f"CheckpointHandler: Checkpoint file {checkpoint_path} does not exist")
