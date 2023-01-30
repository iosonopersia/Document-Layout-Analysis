import os

import torch
from munch import Munch

from tools.checkpoint_handler import CheckpointHandler


def only_if_enabled(func):
    def decorator(*args, **kwargs):
        self = args[0]
        if self.enabled:
            # Execute the function
            return func(*args, **kwargs)
    return decorator


class EarlyStopping:
    def __init__(self, config: Munch, checkpoint_handler: CheckpointHandler) -> None:
        if config is None:
            raise ValueError("EarlyStopping: config cannot be None")
        if checkpoint_handler is None:
            raise ValueError("EarlyStopping: checkpoint_handler cannot be None")

        self.config = config
        self.enabled: bool = config.enabled if hasattr(config, 'enabled') else False
        self.patience: int = config.patience if hasattr(config, 'patience') else 0
        self.restore_best: bool = config.restore_best if hasattr(config, 'restore_best') else False

        self.best_epoch: int = 0
        self.best_val_loss = float('inf')

        if self.restore_best:
            self.checkpoint_handler = checkpoint_handler
            self.best_save_path = os.path.join(os.path.dirname(checkpoint_handler.save_path), f'best_weights.pth')

    @only_if_enabled
    def update(self, epoch: int, val_loss: float, model_state_dict: dict, optimizer_state_dict: dict) -> bool:
        if val_loss < self.best_val_loss:
            self.best_epoch = epoch
            self.best_val_loss = val_loss

            if self.restore_best:
                self.checkpoint_handler.save(
                    epoch,
                    model_state_dict,
                    optimizer_state_dict,
                    save_path=self.best_save_path)

        delta = epoch - self.best_epoch # epochs without improvement
        stop: bool = delta >= self.patience # stop training if patience exceeded
        if stop:
            print(f'EarlyStopping: Stopping training early as no improvement was observed in last {self.patience} epochs.')
            print(f'EarlyStopping: Best validation loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})')

            if self.restore_best:
                best_model = torch.load(self.best_save_path)
                self.checkpoint_handler.save(
                    best_model['epoch'],
                    best_model['model_state_dict'],
                    best_model['optimizer_state_dict'])
                os.remove(self.best_save_path)

        return stop

