"""
Training Pipeline for Traffic Sign Recognition.
Supports training and validating BasicSignClassifier (RawCNN), VGG19, and VGG19-BN
on German (GTSRB), Belgian (BelgiumTSC), and Chinese (CTSD) benchmarks.
Includes upfront training time estimation and user confirmation prompt.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    DATASET_NUM_CLASSES,
    DEVICE,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_WORKERS,
    WEIGHT_DECAY,
)
from data.dataset import get_dataloaders
from evaluation.confusion_matrix import compute_and_save_confusion_matrix
from models.RawCNN import get_model


def format_time(seconds: float) -> str:
    """Format duration in seconds into a clean, human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs:02d}s (~{seconds / 60:.1f} mins)"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins:02d}m (~{seconds / 3600:.2f} hours)"


def estimate_training_time(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    sample_batches: int = 5,
) -> Tuple[float, float]:
    """
    Benchmarks a few sample batches on the target device to accurately estimate:
    - Average time per epoch (training + validation)
    - Total estimated training time across all epochs
    """
    model.train()

    # 1. Warm-up pass (ensures GPU context / CPU threads are warm)
    train_iter = iter(train_loader)
    try:
        w_img, w_tgt = next(train_iter)
        w_img, w_tgt = w_img.to(device), w_tgt.to(device)
        optimizer.zero_grad()
        loss = criterion(model(w_img), w_tgt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    except StopIteration:
        pass

    # 2. Benchmark training batches
    measured_train = 0
    t0_train = time.perf_counter()
    for _ in range(sample_batches):
        try:
            images, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, targets = next(train_iter)

        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        measured_train += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1_train = time.perf_counter()
    avg_train_batch_time = (t1_train - t0_train) / max(measured_train, 1)

    # 3. Benchmark validation batches
    model.eval()
    val_iter = iter(val_loader)
    measured_val = 0
    t0_val = time.perf_counter()
    with torch.no_grad():
        for _ in range(min(sample_batches, len(val_loader))):
            try:
                images, targets = next(val_iter)
            except StopIteration:
                break
            images, targets = images.to(device), targets.to(device)
            _ = model(images)
            measured_val += 1

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1_val = time.perf_counter()
    avg_val_batch_time = (t1_val - t0_val) / max(measured_val, 1)

    # Re-zero gradients and restore train mode
    optimizer.zero_grad()
    model.train()

    # Total batch calculations
    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    est_epoch_time = (n_train_batches * avg_train_batch_time) + (n_val_batches * avg_val_batch_time)
    est_total_time = est_epoch_time * epochs

    return est_epoch_time, est_total_time


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%",
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
    return_preds: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, list, list]]:
    """Evaluate the model on validation or test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if return_preds:
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

    eval_loss = running_loss / total
    eval_acc = 100.0 * correct / total

    if return_preds:
        return eval_loss, eval_acc, all_targets, all_preds
    return eval_loss, eval_acc


def run_training(args):
    print("=" * 65)
    print(f"Traffic Sign Recognition Training: {args.model_type.upper()} on {args.dataset.upper()}")
    print("=" * 65)

    # 1. Resolve number of classes
    num_classes = DATASET_NUM_CLASSES.get(args.dataset.lower(), 43)

    # 2. Data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )
    print(f"Loaded {args.dataset.capitalize()} Dataset ({num_classes} classes):")
    print(f"  Train samples:      {len(train_loader.dataset):,} ({len(train_loader)} batches/epoch)")
    print(f"  Validation samples: {len(val_loader.dataset):,} ({len(val_loader)} batches/epoch)")
    print(f"  Test samples:       {len(test_loader.dataset):,}")

    # 3. Model instantiation
    model = get_model(
        num_classes=num_classes,
        model_type=args.model_type,
        pretrained=args.pretrained,
        compact_head=args.compact_head,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters:     {total_params:,} (Trainable: {trainable_params:,})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Estimate Training Time on Device
    print("\nEstimating training speed on device...")
    est_epoch_time, est_total_time = estimate_training_time(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        epochs=args.epochs,
    )

    print("-" * 65)
    print("  TRAINING TIME ESTIMATION SUMMARY")
    print("-" * 65)
    print(f"  Compute Device:       {DEVICE}")
    print(f"  Model Architecture:   {args.model_type.upper()}")
    print(f"  Target Dataset:       {args.dataset.upper()} ({num_classes} classes)")
    print(f"  Epochs Configured:    {args.epochs}")
    print(f"  Batch Size:           {args.batch_size}")
    print(f"  Estimated / Epoch:    {format_time(est_epoch_time)}")
    print(f"  Estimated Total Time: {format_time(est_total_time)}")
    print("-" * 65)

    # 5. Interactive Confirmation Check
    if not args.yes:
        try:
            proceed = input("\nDo you wish to proceed with training? [Y/n]: ").strip().lower()
            if proceed not in ["y", "yes", ""]:
                print("Training aborted by user.")
                return
        except (KeyboardInterrupt, EOFError):
            print("\nTraining cancelled.")
            return

    # 6. Training Loop
    best_val_acc = 0.0
    save_filename = f"{args.model_type}_{args.dataset}_best.pth"
    save_path = CHECKPOINTS_DIR / save_filename

    print("\nStarting Training Loop...")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE, desc="Validation")
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                    "model_type": args.model_type,
                    "dataset": args.dataset,
                },
                save_path,
            )
            print(f"  --> Saved new best checkpoint to {save_path.name} (Val Acc: {val_acc:.2f}%)")

    total_duration = time.time() - start_time
    print("=" * 65)
    print(f"Training completed in {format_time(total_duration)}.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # 7. Evaluate best checkpoint on test set & Generate Confusion Matrix
    if save_path.exists():
        print("\nLoading best model checkpoint for Test Set Evaluation...")
        checkpoint = torch.load(save_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc, y_true, y_pred = evaluate(
            model, test_loader, criterion, DEVICE, desc="Testing", return_preds=True
        )
        print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.2f}%")

        # Automatically compute and store confusion matrices
        compute_and_save_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            model_name=args.model_type,
            dataset_name=args.dataset,
            split="test",
        )
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Training")
    parser.add_argument(
        "--model_type",
        type=str,
        default="raw_cnn",
        choices=["basic_cnn", "raw_cnn", "cnn", "vgg19", "vgg19_bn"],
        help="Architecture to train ('raw_cnn', 'basic_cnn', 'vgg19', 'vgg19_bn')",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="german",
        choices=["german", "belgian", "chinese"],
        help="Target dataset benchmark",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Explicit path to dataset directory (optional, overrides default paths)",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--img_size", type=int, nargs=2, default=IMAGE_SIZE, help="Image size (H W)")
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS, help="DataLoader workers")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pre-trained weights for VGG")
    parser.add_argument(
        "--compact_head",
        action="store_true",
        default=True,
        help="Use lightweight classification head for VGG19 (reduces params from 143M to 21M)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt and proceed immediately with training",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
