import os
os.environ["MAMBA_FORCE_FALLBACK"] = "1"

import csv
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel


if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    @property
    def all_tied_weights_keys(self):
        tied = getattr(self, "_tied_weights_keys", None)
        if isinstance(tied, dict):
            return tied
        if isinstance(tied, (list, tuple, set)):
            return {k: None for k in tied}
        return {}

    PreTrainedModel.all_tied_weights_keys = all_tied_weights_keys

# =========================
# CONFIG
# =========================
train_dir = "/home/teaching/dl_mamba/data_real_split/train"
test_dir = "/home/teaching/dl_mamba/data_real_split/test"

batch_size = 16
num_epochs = 25
lr = 1e-4
step_size = 8
gamma = 0.5
best_model_path = "best_mamba_model.pth"
loss_curve_path = "loss_curve.png"
accuracy_curve_path = "accuracy_curve.png"
combined_curve_path = "training_curves.png"
metrics_csv_path = "training_metrics.csv"
metrics_json_path = "training_summary.json"

# Count classes
num_classes = len(datasets.ImageFolder(train_dir).classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Number of classes: {num_classes}")

# =========================
# DATA
# =========================
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

loader_kwargs = {
    "batch_size": batch_size,
    "num_workers": 4,
    "pin_memory": torch.cuda.is_available(),
}

train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(test_dataset)}")

# =========================
# MODEL
# =========================
class MambaClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "nvidia/MambaVision-T-1K",
            trust_remote_code=True,
            ignore_mismatched_sizes=True
        )
        if hasattr(self.backbone.config, "tie_word_embeddings"):
            self.backbone.config.tie_word_embeddings = False
        self.classifier = nn.LazyLinear(num_classes)

    @staticmethod
    def _extract_features(outputs):
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state")
            if hidden is None:
                hidden = outputs.get("features")
            if hidden is None and len(outputs) > 0:
                hidden = next(iter(outputs.values()))
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            hidden = outputs[0]
        else:
            hidden = outputs

        if isinstance(hidden, (tuple, list)) and len(hidden) > 0:
            hidden = hidden[0]

        if hidden.dim() == 4:
            return hidden.mean(dim=(2, 3))
        if hidden.dim() == 3:
            return hidden.mean(dim=1)
        if hidden.dim() == 2:
            return hidden
        return hidden.flatten(start_dim=1)

    def forward(self, x):
        outputs = self.backbone(x)
        features = self._extract_features(outputs)
        return self.classifier(features)

model = MambaClassifier(num_classes)
model.to(device)

# =========================
# TRAIN SETUP
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def move_batch_to_device(images, labels, device):
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = move_batch_to_device(images, labels, device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size_local = labels.size(0)
            total_loss += loss.item() * batch_size_local
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size_local

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def save_metrics_csv(csv_path, epochs, train_losses, val_losses, val_accs, lrs):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_acc"])
        for e, lr_val, tr, va_l, va_a in zip(epochs, lrs, train_losses, val_losses, val_accs):
            writer.writerow([e, f"{lr_val:.8f}", f"{tr:.8f}", f"{va_l:.8f}", f"{va_a:.8f}"])


def save_summary_json(json_path, best_acc, best_epoch, epochs, train_losses, val_losses, val_accs, lrs):
    payload = {
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "last_epoch": epochs[-1] if epochs else None,
        "last_train_loss": train_losses[-1] if train_losses else None,
        "last_val_loss": val_losses[-1] if val_losses else None,
        "last_val_acc": val_accs[-1] if val_accs else None,
        "history": [
            {
                "epoch": e,
                "lr": lr_val,
                "train_loss": tr,
                "val_loss": va_l,
                "val_acc": va_a,
            }
            for e, lr_val, tr, va_l, va_a in zip(epochs, lrs, train_losses, val_losses, val_accs)
        ],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


def save_plots(epoch_history, train_loss_history, val_loss_history, val_acc_history):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Warning: matplotlib is not installed; skipping curve generation.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_history, train_loss_history, label="Train Loss", marker="o")
    plt.plot(epoch_history, val_loss_history, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs Train/Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to {loss_curve_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_history, val_acc_history, label="Validation Accuracy", color="green", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Epoch vs Validation Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(accuracy_curve_path, dpi=200)
    plt.close()
    print(f"Saved accuracy curve to {accuracy_curve_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epoch_history, train_loss_history, label="Train Loss", marker="o")
    axes[0].plot(epoch_history, val_loss_history, label="Validation Loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epoch_history, val_acc_history, label="Validation Accuracy", color="green", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Training Curves")
    fig.tight_layout()
    fig.savefig(combined_curve_path, dpi=200)
    plt.close(fig)
    print(f"Saved combined curves to {combined_curve_path}")

# =========================
# TRAIN LOOP
# =========================
best_val_acc = 0.0
best_epoch = 0
train_loss_history = []
val_loss_history = []
val_acc_history = []
epoch_history = []
lr_history = []

print(f"Metrics CSV will be written to {metrics_csv_path}")
print(f"Summary JSON will be written to {metrics_json_path}")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_samples = 0

    for images, labels in train_loader:
        images, labels = move_batch_to_device(images, labels, device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size_local = labels.size(0)
        running_loss += loss.item() * batch_size_local
        running_samples += batch_size_local

    train_loss = running_loss / max(running_samples, 1)
    val_loss, val_acc = evaluate(model, test_loader, criterion, device)
    current_lr = optimizer.param_groups[0]["lr"]

    epoch_number = epoch + 1
    epoch_history.append(epoch_number)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)
    lr_history.append(current_lr)

    # Persist metrics every epoch so progress is saved even if training stops early.
    save_metrics_csv(metrics_csv_path, epoch_history, train_loss_history, val_loss_history, val_acc_history, lr_history)

    print(
        f"Epoch [{epoch_number}/{num_epochs}] "
        f"LR: {current_lr:.6f} "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch_number
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved new best model to {best_model_path} (val acc: {best_val_acc:.4f})")

    scheduler.step()

# =========================
# SAVE MODEL
# =========================
if not os.path.exists(best_model_path):
    torch.save(model.state_dict(), best_model_path)
    print(f"Saved final model to {best_model_path}")

save_summary_json(
    metrics_json_path,
    best_val_acc,
    best_epoch,
    epoch_history,
    train_loss_history,
    val_loss_history,
    val_acc_history,
    lr_history,
)
print(f"Saved training summary to {metrics_json_path}")

save_plots(epoch_history, train_loss_history, val_loss_history, val_acc_history)

print(f"✅ Training complete. Best validation accuracy: {best_val_acc:.4f}")