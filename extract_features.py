import os
os.environ["MAMBA_FORCE_FALLBACK"] = "1"

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
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
data_dir = "/home/teaching/dl_mamba/data_real_split"
model_path = "best_mamba_model.pth"
batch_size = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# TRANSFORM (same as training)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =========================
# DATASET
# =========================
dataset = datasets.ImageFolder("/home/teaching/dl_mamba/data_real", transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# =========================
# MODEL (same structure)
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

    def forward(self, x):
        outputs = self.backbone(x)

        # handle different outputs
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state
        elif isinstance(outputs, dict):
            hidden = outputs.get("last_hidden_state")
            if hidden is None and len(outputs) > 0:
                hidden = next(iter(outputs.values()))
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            hidden = outputs[0]
        else:
            hidden = outputs

        if isinstance(hidden, (tuple, list)) and len(hidden) > 0:
            hidden = hidden[0]

        if hidden.dim() == 4:
            features = hidden.mean(dim=(2, 3))
        elif hidden.dim() == 3:
            features = hidden.mean(dim=1)
        elif hidden.dim() == 2:
            features = hidden
        else:
            features = hidden.flatten(start_dim=1)

        return features  # IMPORTANT: return features (not logits)

# =========================
# LOAD MODEL
# =========================
num_classes = len(dataset.classes)
model = MambaClassifier(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")

# =========================
# FEATURE EXTRACTION
# =========================
all_features = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        features = model(images)  # [batch, feature_dim]
        features = features.cpu()

        all_features.append(features)
        all_labels.append(labels)

# =========================
# CONCATENATE
# =========================
all_features = torch.cat(all_features)
all_labels = torch.cat(all_labels)

print("Feature extraction complete!")
print("Features shape:", all_features.shape)

# =========================
# SAVE
# =========================
torch.save({
    "features": all_features,
    "labels": all_labels
}, "features.pt")

print("Saved features to features.pt")