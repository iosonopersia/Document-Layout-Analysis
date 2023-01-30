import os
from datetime import datetime
from typing import Any

from munch import Munch

import wandb


def only_if_enabled(func):
    def decorator(*args, **kwargs):
        self = args[0]
        if self.enabled:
            # Execute the function
            return func(*args, **kwargs)
    return decorator


class WandBLogger():
    def __init__(self, config: Munch) -> None:
        if config is None:
            raise ValueError("WandBLogger: config cannot be None")

        self.config = config
        self.enabled: bool = config.enabled if hasattr(config, 'enabled') else False
        self.entity_name: str = config.entity if hasattr(config, 'entity') else None
        self.project_name: str = config.project if hasattr(config, 'project') else None
        self.resume_run: bool = config.resume_run if hasattr(config, 'resume_run') else False
        self.resume_run_id: str = config.resume_run_id if hasattr(config, 'resume_run_id') else None
        self.watch_model: bool = config.watch_model if hasattr(config, 'watch_model') else False
        self.watch_model_type: str = config.watch_model_type if hasattr(config, 'watch_model_type') else None

        self._step: int = 0

    @only_if_enabled
    def step(self) -> None:
        self._step += 1

    @only_if_enabled
    def reset_step(self) -> None:
        self._step = 0

    @only_if_enabled
    def start_new_run(self, run_config: Munch = None) -> None:
        self.reset_step()

        os_start_method = 'spawn' if os.name == 'nt' else 'fork'
        run_datetime = datetime.now().isoformat().split('.')[0]
        wandb.init(
            project=self.project_name,
            name=run_datetime,
            config=run_config,
            settings=wandb.Settings(start_method=os_start_method),
            entity=self.entity_name,
            resume="must" if self.resume_run else False,
            id=self.resume_run_id)

    @only_if_enabled
    def log(self, metrics: dict) -> None:
        wandb.log(metrics, step=self._step)

    @only_if_enabled
    def stop_run(self) -> None:
        wandb.finish()
        self.reset_step()

    @only_if_enabled
    def start_watcher(self, model: Any, criterion: Any, log_freq: int = 1) -> None:
        if self.watch_model:
            wandb.watch(
                model,
                criterion=criterion,
                log=self.watch_model_type,
                log_freq=log_freq,
                log_graph=False)
