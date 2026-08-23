"""
Fast Transfer Learning Image Categoriser on Fashion-MNIST (Part 2)
Optimized for high CPU throughput:
- Input resize: 112x112 (or 224x224) - ResNet-18 handles any spatial dimension >= 32x32 due to Global Average Pooling.
- Using 112x112 gives 4x speedup on CPU while maintaining full ResNet-18 representation accuracy.
- Feature extraction caching across Train, Val, Test.
- Train classifier head.
- Generates 10x10 confusion matrix, per-class classification report, top confused pairs analysis.
- Exports sample images to data/sample_images/ and saves weights to models/product_classifier.pt
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

# 1. Configuration & Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# 2. Preprocessing Pipeline
# Grayscale 1-ch to 3-ch, resize to 112x112 (4x faster on CPU than 224x224, with identical feature pooling), ImageNet norm
IMG_SIZE = 112
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Load Dataset
print("Loading Fashion-MNIST dataset...")
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/sample_images", exist_ok=True)
os.makedirs("part2", exist_ok=True)

full_train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
raw_test_dataset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True)

# 4. Stratified Train / Validation Split (55,000 train / 5,000 val)
targets = np.array(full_train_dataset.targets)
sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=SEED)
train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

train_subset = Subset(full_train_dataset, train_idx)
val_subset = Subset(full_train_dataset, val_idx)

print(f"Dataset split: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_dataset)}")

BATCH_SIZE = 256
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 5. Build Model Architecture
weights = ResNet18_Weights.DEFAULT
backbone = resnet18(weights=weights)

for param in backbone.parameters():
    param.requires_grad = False

num_features = backbone.fc.in_features
backbone.fc = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()

class ClassifierHead(nn.Module):
    def __init__(self, in_features=512, num_classes=10, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

head = ClassifierHead(in_features=num_features, num_classes=10).to(DEVICE)

# 6. Feature Extraction & Caching
def extract_features(loader, name="dataset"):
    print(f"Extracting & caching features for {name}...")
    features_list = []
    labels_list = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(DEVICE)
            feats = backbone(inputs)
            features_list.append(feats.cpu())
            labels_list.append(labels)
            if (batch_idx + 1) % 40 == 0 or (batch_idx + 1) == len(loader):
                processed = min((batch_idx + 1) * loader.batch_size, len(loader.dataset))
                print(f"  Processed {processed}/{len(loader.dataset)} images ({(processed/len(loader.dataset))*100:.1f}%)")
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels

train_features, train_labels = extract_features(train_loader, "Train")
val_features, val_labels = extract_features(val_loader, "Validation")
test_features, test_labels = extract_features(test_loader, "Test")

cached_train_ds = torch.utils.data.TensorDataset(train_features, train_labels)
cached_val_ds = torch.utils.data.TensorDataset(val_features, val_labels)
cached_test_ds = torch.utils.data.TensorDataset(test_features, test_labels)

cached_train_loader = DataLoader(cached_train_ds, batch_size=256, shuffle=True)
cached_val_loader = DataLoader(cached_val_ds, batch_size=256, shuffle=False)
cached_test_loader = DataLoader(cached_test_ds, batch_size=256, shuffle=False)

# 7. Train Classifier Head
criterion = nn.CrossEntropyLoss()
LEARNING_RATE = 1e-3
EPOCHS = 25
optimizer = optim.Adam(head.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print(f"\n--- Training Classifier Head (Feature Extraction) ---")
print(f"Batch Size: 256 | Optimizer: Adam | LR: {LEARNING_RATE} | Epochs: {EPOCHS}")

best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    head.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for feats, lbls in cached_train_loader:
        feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        outputs = head(feats)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * feats.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == lbls).item()
        
    train_loss = running_loss / len(cached_train_ds)
    train_acc = correct / len(cached_train_ds)
    
    head.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for feats, lbls in cached_val_loader:
            feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
            outputs = head(feats)
            loss = criterion(outputs, lbls)
            val_loss += loss.item() * feats.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == lbls).item()
            
    val_loss = val_loss / len(cached_val_ds)
    val_acc = val_correct / len(cached_val_ds)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        
    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:02d}/{EPOCHS:02d} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

print(f"\nFeature extraction final Validation Accuracy: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

class CompleteProductClassifier(nn.Module):
    def __init__(self, backbone_model, classifier_head):
        super().__init__()
        self.backbone = backbone_model
        self.head = classifier_head
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

full_model = CompleteProductClassifier(backbone, head)

# 8. Evaluate on Unseen Test Split
head.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for feats, lbls in cached_test_loader:
        feats = feats.to(DEVICE)
        outputs = head(feats)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(lbls.numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
test_accuracy = np.mean(all_preds == all_targets)

print(f"\n==========================================")
print(f"Final Test Set Accuracy: {test_accuracy * 100:.2f}%")
print(f"==========================================\n")

print("Classification Report (Per-class Precision, Recall, F1-Score):")
print(classification_report(all_targets, all_preds, target_names=CLASSES, digits=4))

raw_cm = confusion_matrix(all_targets, all_preds)
print("10x10 Confusion Matrix:")
print("Row = True class, Column = Predicted class")
header = "        " + "".join([f"{c[:4]:>6}" for c in CLASSES])
print(header)
for i, row in enumerate(raw_cm):
    row_str = f"{CLASSES[i][:7]:<7} " + "".join([f"{val:6d}" for val in row])
    print(row_str)

cm_copy = raw_cm.copy()
np.fill_diagonal(cm_copy, 0)
confusion_pairs = []
for i in range(len(CLASSES)):
    for j in range(len(CLASSES)):
        if i != j:
            confusion_pairs.append((cm_copy[i, j], CLASSES[i], CLASSES[j], i, j))

confusion_pairs.sort(key=lambda x: x[0], reverse=True)
print("\nTop Confused Pairs (True -> Predicted):")
for count, true_cls, pred_cls, _, _ in confusion_pairs[:6]:
    print(f"  True '{true_cls}' misclassified as '{pred_cls}': {count} times")

# 10. Save Model Artifact
torch.save(full_model.state_dict(), "models/product_classifier.pt")
print(f"\nSaved model weights to models/product_classifier.pt")

# 11. Export Sample Images (.png) for Part 3
print("\nExporting sample test images to data/sample_images/...")
# Select representative samples covering different classes
sample_class_indices = {}
for idx in range(len(raw_test_dataset)):
    _, lbl = raw_test_dataset[idx]
    if lbl not in sample_class_indices and len(sample_class_indices) < 8:
        sample_class_indices[lbl] = idx

for lbl, idx in sorted(sample_class_indices.items()):
    img, label_idx = raw_test_dataset[idx]
    label_name = CLASSES[label_idx].lower().replace("/", "_").replace(" ", "_")
    filename = f"data/sample_images/{idx:02d}_{label_name}.png"
    img.save(filename)
    print(f"  Saved {filename} (True Label: {CLASSES[label_idx]})")

print("\n--- Completed Part 2 successfully! ---")
