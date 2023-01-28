import os
from collections import OrderedDict
from datetime import datetime

import torch
import torch.optim as optim
import wandb

from tqdm import tqdm

from dataset import get_data
from loss import YoloLoss
from model import DocumentObjectDetector
from utils import get_anchors_dict, get_config

torch.backends.cudnn.benchmark = True # improves training speed if input size doesn't change


def train_fn(train_loader, model, optimizer, epoch, loss_fn):
    loop = tqdm(train_loader, leave=True)
    loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
    loop.set_postfix(loss=0.0)

    accumulation_steps = hyperparams_cfg.gradient_accumulation_steps
    loss_value = 0.0
    loss = torch.zeros(1, requires_grad=True)

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
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward() # accumulate gradients
        if ((i + 1) % accumulation_steps == 0) or ((i + 1) == num_batches):
            # Update parameters
            optimizer.step()
            # Reset gradients
            model.zero_grad()

        # Update progress bar
        loss_value += loss.item() * accumulation_steps
        loop.set_postfix(loss=loss_value / (i+1))
        if wandb_cfg.enabled:
            wandb.log({'Train/Running_loss': loss_value / (i+1)})

    return loss_value / num_batches


def eval_fn(eval_loader, model, loss_fn):
    loop = tqdm(eval_loader, leave=True)
    loop.set_description(f"Validation")
    loop.set_postfix(loss=0.0)

    loss_value = 0.0

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
            loss_value += loss.item()

            # Update progress bar
            loop.set_postfix(loss=loss_value / (i+1))

    return loss_value / num_batches


def train_loop():
    # ========== DATASET=============
    train_dataset, train_loader = get_data("train")
    eval_dataset, eval_loader = get_data("val")

    # ===========MODEL===============
    model = DocumentObjectDetector()
    model = model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    #============WANDB===============
    if wandb_cfg.enabled:
        entity_name = wandb_cfg.entity
        resume_run = wandb_cfg.resume_run
        resume_run_id = wandb_cfg.resume_run_id

        os_start_method = 'spawn' if os.name == 'nt' else 'fork'
        run_datetime = datetime.now().isoformat().split('.')[0]
        wandb.init(
            project=wandb_cfg.project_name,
            name=run_datetime,
            config=config,
            settings=wandb.Settings(start_method=os_start_method),
            entity = entity_name,
            resume="must" if resume_run else False,
            id=resume_run_id)

        if wandb_cfg.watch_model:
            wandb.watch(
                model,
                criterion=loss_fn,
                log="all", # default("gradients"), "parameters", "all"
                log_freq=1,
                log_graph=False)

    #===========CHECKPOINT===========
    checkpoint_cfg = config.checkpoint
    save_checkpoint = checkpoint_cfg.save_checkpoint
    save_checkpoint_path = os.path.abspath(checkpoint_cfg.save_path)
    os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True) # Create folder if it doesn't exist

    start_epoch = 0
    load_checkpoint = checkpoint_cfg.load_checkpoint
    load_checkpoint_path = os.path.abspath(checkpoint_cfg.load_path)

    if (load_checkpoint and os.path.exists(load_checkpoint_path)):
        checkpoint = torch.load(load_checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1 # start from the next epoch
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # ========EARLY STOPPING=========
    early_stopping_cfg = hyperparams_cfg.early_stopping
    if early_stopping_cfg.enabled:
        best_weights_save_path = os.path.join(os.path.dirname(save_checkpoint_path), f'best_weights.pth')
        min_loss_val = float('inf')
        patience_counter = 0

    # ===========TRAINING============
    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_fn(train_loader, model, optimizer, epoch, loss_fn)
        val_loss = eval_fn(eval_loader, model, loss_fn)

        # ============WANDB==============
        if wandb_cfg.enabled:
            wandb.log({
                'Params/Epoch': epoch,
                'Train/Loss': train_loss,
                'Validation/Loss': val_loss,
            })

        # ===========CHECKPOINT==========
        if save_checkpoint:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, save_checkpoint_path)

        # ========EARLY STOPPING=========
        if early_stopping_cfg.enabled:
            if val_loss < min_loss_val:
                min_loss_val = val_loss
                patience_counter = 0
                if early_stopping_cfg.restore_best_weights:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                    }, best_weights_save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_cfg.patience:
                    print(f"Early stopping: patience {early_stopping_cfg.patience} reached")
                    if early_stopping_cfg.restore_best_weights:
                        best_model = torch.load(best_weights_save_path)
                        torch.save({
                            'epoch': best_model['epoch'],
                            'model_state_dict': best_model['model_state_dict'],
                        }, save_checkpoint_path)
                        os.remove(best_weights_save_path)
                        print(f"Best model at epoch {best_model['epoch']} restored")
                    break

    if wandb_cfg.enabled:
        wandb.finish()


if __name__ == "__main__":
    #============DEVICE===============
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Using device {DEVICE}]")

    # ===========CONFIG===============
    config = get_config()
    dataset_cfg = config.dataset
    model_cfg = config.model
    hyperparams_cfg = config.hyperparameters.train
    dataloader_cfg = config.dataloader.train
    wandb_cfg = config.wandb

    # ===========DATASET==============
    ANCHORS_DICT = get_anchors_dict(dataset_cfg.anchors_file)
    SCALED_ANCHORS = OrderedDict({
        size: torch.tensor(anchors, dtype=torch.float32, device='cpu') * size
        for size, anchors in ANCHORS_DICT.items()
    })

    # ========HYPERPARAMETERS========
    LR = hyperparams_cfg.learning_rate
    WEIGHT_DECAY = hyperparams_cfg.weight_decay
    EPOCHS = hyperparams_cfg.epochs
    ACCUMULATION_STEPS = hyperparams_cfg.gradient_accumulation_steps

    train_loop()
