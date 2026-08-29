import json
from pathlib import Path

import torch
import yaml
from torch import nn

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: dict,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Build a learning-rate scheduler from config.
    Supports: cosine, step, none.
    """
    stype = scheduler_cfg.get("type", "none").lower()

    if stype == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg.get("T_max", epochs),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
    if stype == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 5),
            gamma=scheduler_cfg.get("gamma", 0.1),
        )
    return None


def main() -> None:
    # Config resolution: K8s volume mount takes priority, falls back to local path
    config_path = Path("/app/configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")

    config = load_config(str(config_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "startup", "device": str(device)}), flush=True)

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"].get("num_workers", 2),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )

    scheduler = build_scheduler(
        optimizer,
        config["training"].get("lr_scheduler", {}),
        epochs=config["training"]["epochs"],
    )

    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / config["output"]["model_name"]

    # Track best by val accuracy (more stable than val loss with LR scheduling)
    best_val_acc = 0.0
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "learning_rate": round(current_lr, 7),
        }
        print(json.dumps(log_entry), flush=True)

        # Step scheduler after logging so the logged LR matches the epoch
        if scheduler is not None:
            scheduler.step()

        # Save on best val accuracy (not val loss) to be robust to LR-driven loss dips
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "config": config,
                },
                save_path,
            )
            print(
                json.dumps({"event": "checkpoint_saved", "path": str(save_path), "val_accuracy": round(val_acc, 4)}),
                flush=True,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    json.dumps({"event": "early_stopping", "epoch": epoch + 1, "best_val_accuracy": round(best_val_acc, 4)}),
                    flush=True,
                )
                break

    print(
        json.dumps({"event": "training_complete", "best_val_accuracy": round(best_val_acc, 4)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
