from collections import OrderedDict

import torch
import torch.optim as optim
from tqdm import tqdm

from dataset import get_dataloader
from loss import YoloLoss
from model import DocumentObjectDetector
from tools.checkpoint_handler import CheckpointHandler
from tools.early_stopping import EarlyStopping
from tools.wandb_logger import WandBLogger
from utils import get_anchors_dict, get_config

torch.backends.cudnn.benchmark = True # improves training speed if input size doesn't change


def train_fn(train_loader, model, optimizer, epoch, loss_fn):
    loop = tqdm(train_loader, leave=True)
    loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
    loop.set_postfix(loss=0.0)

    gradient_clip_cfg = hyperparams_cfg.gradient_clip
    epoch_loss = 0.0

    model.train()
    model.zero_grad()
    num_batches = len(train_loader)
    for i, batch in enumerate(loop):
        images = batch['image'].to(DEVICE)
        targets = batch['target']

        # Forward pass
        predictions = model(images)
        for key in predictions.keys():
            predictions[key] = predictions[key].cpu()

        # Loss computation
        loss = loss_fn(predictions, targets, SCALED_ANCHORS)

        # Backward pass
        (loss / ACCUMULATION_STEPS).backward() # accumulate gradients
        epoch_loss += loss.item()

        if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == num_batches):
            # Gradient clipping
            if gradient_clip_cfg.enabled:
                torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=gradient_clip_cfg.max_grad_norm,
                    norm_type=gradient_clip_cfg.grad_norm_type)
            # Update parameters
            optimizer.step()
            # Reset gradients
            model.zero_grad()

            # Update progress bar
            loop.set_postfix(loss=epoch_loss / (i+1))

            wandb_logger.log({'Train/Running_loss': epoch_loss / (i+1)})
            wandb_logger.step()

    return epoch_loss / num_batches


def eval_fn(eval_loader, model, loss_fn):
    loop = tqdm(eval_loader, leave=True)
    loop.set_description(f"Validation")
    loop.set_postfix(loss=0.0)

    eval_loss = 0.0

    model.eval()
    num_batches = len(eval_loader)
    with torch.inference_mode():
        for i, batch in enumerate(loop):
            images = batch['image'].to(DEVICE)
            targets = batch['target']

            # Forward pass
            predictions = model(images)
            for key in predictions.keys():
                predictions[key] = predictions[key].cpu()
            loss = loss_fn(predictions, targets, SCALED_ANCHORS)
            eval_loss += loss.item()

            # Update progress bar
            loop.set_postfix(loss=eval_loss / (i+1))

    return eval_loss / num_batches


def train_loop():
    # ========== DATASET=============
    train_loader = get_dataloader("train")
    eval_loader = get_dataloader("val")

    # ===========MODEL===============
    model = DocumentObjectDetector(NUM_CLASSES, model_cfg)
    model = model.to(DEVICE)
    loss_fn = YoloLoss()

    # ===========OPTIMIZER===========
    model.freeze_backbone() # marks backbone parameters as not trainable
    frozen_optimizer = optim.SGD(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=FROZEN_LR,
        momentum=MOMENTUM,
        weight_decay=WD,
        nesterov=True)

    model.unfreeze_backbone() # marks all parameters as trainable
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WD,
        nesterov=True)

    #============WANDB===============
    wandb_logger.start_new_run(run_config=config)
    wandb_logger.start_watcher(
        model=model,
        criterion=loss_fn,
        log_freq=len(train_loader) // ACCUMULATION_STEPS)

    #===========CHECKPOINT===========
    start_epoch = checkpoint_handler.restore(model, optimizer)

    # ===========TRAINING============
    for epoch in range(start_epoch, EPOCHS):
        is_frozen_epoch: bool = epoch < FROZEN_EPOCHS
        if is_frozen_epoch:
            model.freeze_backbone()
            epoch_optimizer = frozen_optimizer
        else:
            model.unfreeze_backbone()
            epoch_optimizer = optimizer

        train_loss = train_fn(train_loader, model, epoch_optimizer, epoch, loss_fn)
        val_loss = eval_fn(eval_loader, model, loss_fn)

        # ============WANDB==============
        wandb_logger.log({
            'Params/Epoch': epoch,
            'Train/Loss': train_loss,
            'Validation/Loss': val_loss,
        })

        # ===========CHECKPOINT==========
        model_state_dict = model.state_dict()
        optimizer_state_dict = epoch_optimizer.state_dict()
        checkpoint_handler.save(epoch, model_state_dict, optimizer_state_dict, is_frozen_epoch)

        # ========EARLY STOPPING=========
        stop: bool = early_stopping.update(epoch, val_loss, model_state_dict, optimizer_state_dict, is_frozen_epoch)
        if stop:
            break

    wandb_logger.stop_run()


if __name__ == "__main__":
    #============DEVICE===============
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Using device {DEVICE}]")

    # ===========CONFIG===============
    config = get_config()
    dataset_cfg = config.dataset
    model_cfg = config.model
    hyperparams_cfg = config.hyperparameters
    finetuning_cfg = hyperparams_cfg.finetuning

    # ============TOOLS==============
    wandb_logger = WandBLogger(config.wandb)
    checkpoint_handler = CheckpointHandler(config.checkpoint)
    early_stopping = EarlyStopping(hyperparams_cfg.early_stopping, checkpoint_handler)

    # ===========DATASET==============
    NUM_CLASSES = dataset_cfg.num_classes
    ANCHORS_DICT = get_anchors_dict(dataset_cfg.anchors_file)
    SCALED_ANCHORS = OrderedDict({
        size: anchors * size
        for size, anchors in ANCHORS_DICT.items()
    })

    # ========HYPERPARAMETERS========
    FROZEN_EPOCHS = finetuning_cfg.frozen_epochs
    FROZEN_LR = finetuning_cfg.frozen_learning_rate

    LR = hyperparams_cfg.learning_rate
    WD = hyperparams_cfg.weight_decay
    MOMENTUM = hyperparams_cfg.momentum
    EPOCHS = hyperparams_cfg.epochs
    ACCUMULATION_STEPS = hyperparams_cfg.gradient_accumulation_steps

    train_loop()
