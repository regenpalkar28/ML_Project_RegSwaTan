"""
Training Pipeline for Traffic Sign Recognition.
Supports training and validating BasicSignClassifier (CNN), VGG19, and VGG19-BN
on German (GTSRB), Belgian (BelgiumTSC), and Chinese (CTSD) benchmarks.
"""

import argparse
import sys
import time
from pathlib import Path

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
from models.model import get_model


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
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
) -> tuple:
    """Evaluate the model on validation or test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, targets in pbar:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    eval_loss = running_loss / total
    eval_acc = 100.0 * correct / total
    return eval_loss, eval_acc


def run_training(args):
    print("=" * 65)
    print(f"Starting Training: Model={args.model_type.upper()} | Dataset={args.dataset.upper()}")
    print(f"Device: {DEVICE} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 65)

    # Resolve number of classes
    num_classes = DATASET_NUM_CLASSES.get(args.dataset.lower(), 43)

    # Data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )
    print(f"Loaded {args.dataset} dataset:")
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Val samples:   {len(val_loader.dataset):,}")
    print(f"  Test samples:  {len(test_loader.dataset):,}")

    # Model instantiation
    model = get_model(
        num_classes=num_classes,
        model_type=args.model_type,
        pretrained=args.pretrained,
        compact_head=args.compact_head,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    save_filename = f"{args.model_type}_{args.dataset}_best.pth"
    save_path = CHECKPOINTS_DIR / save_filename

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

    total_duration = (time.time() - start_time) / 60.0
    print("=" * 65)
    print(f"Training completed in {total_duration:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    # Evaluate best checkpoint on test set
    if save_path.exists():
        print("\nLoading best model checkpoint for Test Set Evaluation...")
        checkpoint = torch.load(save_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE, desc="Testing")
        print(f"Final Test Loss: {test_loss:.4f} | Final Test Accuracy: {test_acc:.2f}%")
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Sign Recognition Training")
    parser.add_argument(
        "--model_type",
        type=str,
        default="basic_cnn",
        choices=["basic_cnn", "vgg19", "vgg19_bn"],
        help="Architecture to train ('basic_cnn', 'vgg19', 'vgg19_bn')",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
