"""
=======================================================================
HRI Motion LSTM Trainer v2
=======================================================================
Train a PyTorch LSTM model on the v2 synthetic motion dataset.

Architecture:
  - Input:  99 features (33 keypoints × 3 coords)
  - LSTM:   3 layers, hidden_size=128, dropout=0.4
  - FC:     128 → 64 → 32 → 9 classes
  - Seq:    29 velocity steps (from 30 frames)

Usage:
  python 2_train_and_evaluate_v2.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────
KEYPOINTS_DIR = Path("extracted_keypoints_v2")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Hyperparameters
INPUT_SIZE = 99        # 33 keypoints × 3 coords
HIDDEN_SIZE = 128
NUM_LAYERS = 3
DROPOUT = 0.4
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
VELOCITY_SCALE = 100.0

# Random seeds
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# ──────────────────────────────────────────────────────────────────────
# LOAD DATASET
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Loading dataset...")
print("=" * 60)

with open("dataset_info_v2.json", "r") as f:
    dataset_config = json.load(f)

MOTION_LABELS = dataset_config["motion_labels"]
NUM_CLASSES = len(MOTION_LABELS)

print(f"\nMotion classes: {NUM_CLASSES}")
for i, label in enumerate(MOTION_LABELS):
    print(f"  {i}: {label}")

sequences = []
labels = []

for motion_dir in sorted(KEYPOINTS_DIR.iterdir()):
    if not motion_dir.is_dir():
        continue
    motion_id = int(motion_dir.name.split("_")[0])
    npy_files = sorted(motion_dir.glob("*.npy"))
    for npy_file in tqdm(npy_files, desc=MOTION_LABELS[motion_id], leave=True):
        keypoints = np.load(npy_file)
        sequences.append(keypoints)
        labels.append(motion_id)

print(f"\nTotal sequences loaded: {len(sequences)}")
print(f"Sample shape: {sequences[0].shape}")


# ──────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — VELOCITY COMPUTATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Computing velocity features...")
print("=" * 60)

def compute_velocity_features(keypoints):
    """Convert keypoint positions to scaled velocity vectors."""
    velocities = np.diff(keypoints, axis=0)
    return velocities * VELOCITY_SCALE

velocity_sequences = []
for seq in tqdm(sequences, desc="Velocities"):
    vel = compute_velocity_features(seq)
    velocity_sequences.append(vel)

frame_lengths = [v.shape[0] for v in velocity_sequences]
MAX_LEN = int(np.percentile(frame_lengths, 90))
print(f"\nVelocity sequence shape: {velocity_sequences[0].shape}")
print(f"Max sequence length: {MAX_LEN}")

# Velocity statistics
all_vel = np.concatenate([v.reshape(-1) for v in velocity_sequences])
print(f"\nVelocity statistics (after ×{VELOCITY_SCALE} scaling):")
print(f"  Mean: {all_vel.mean():.4f}")
print(f"  Std:  {all_vel.std():.4f}")
print(f"  Min:  {all_vel.min():.4f}")
print(f"  Max:  {all_vel.max():.4f}")


# ──────────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────────────
class MotionDataset(Dataset):
    def __init__(self, sequences, labels, max_len):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        if seq.shape[0] < self.max_len:
            pad_len = self.max_len - seq.shape[0]
            seq = np.pad(seq, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
        else:
            seq = seq[:self.max_len]
        return torch.FloatTensor(seq), torch.LongTensor([label])[0]


# Train/test split (80/20)
train_idx, test_idx = train_test_split(
    range(len(velocity_sequences)),
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

train_sequences = [velocity_sequences[i] for i in train_idx]
train_labels = [labels[i] for i in train_idx]
test_sequences = [velocity_sequences[i] for i in test_idx]
test_labels = [labels[i] for i in test_idx]

train_dataset = MotionDataset(train_sequences, train_labels, MAX_LEN)
test_dataset = MotionDataset(test_sequences, test_labels, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"\nTrain set: {len(train_dataset)} sequences")
print(f"Test set:  {len(test_dataset)} sequences")
print(f"Batch size: {BATCH_SIZE}")

train_counts = Counter(train_labels)
test_counts = Counter(test_labels)
print("\nClass distribution:")
for i in range(NUM_CLASSES):
    print(f"  {i}: {MOTION_LABELS[i]:25s} — Train: {train_counts[i]:4d}, Test: {test_counts[i]:3d}")


# ──────────────────────────────────────────────────────────────────────
# LSTM MODEL
# ──────────────────────────────────────────────────────────────────────
class MotionLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=128, num_layers=3, num_classes=9, dropout=0.4):
        super(MotionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        x = x.reshape(batch_size, seq_len, -1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return out


model = MotionLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT,
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "=" * 60)
print("Model Architecture:")
print("=" * 60)
print(model)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# ──────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

best_val_loss = float("inf")
best_val_acc = 0.0
best_epoch = 0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print("\n" + "=" * 60)
print("Starting training...")
print("=" * 60)
print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>10}  {'ValLoss':>10}  {'ValAcc':>10}  {'LR':>10}")
print("—" * 66)

for epoch in range(1, EPOCHS + 1):
    # ── Train ──
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_x.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ── Validate ──
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_running_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == batch_y).sum().item()
            val_total += batch_y.size(0)

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), MODELS_DIR / "motion_lstm_v2_best.pth")

    if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.2%}  {val_loss:>10.4f}  {val_acc:>9.2%}  {current_lr:>10.6f}")

# Save final model
torch.save(model.state_dict(), MODELS_DIR / "motion_lstm_v2_final.pth")

print(f"\n[SUCCESS] Best model at epoch {best_epoch}: val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.2%}")
print(f"[SUCCESS] Saved: {MODELS_DIR / 'motion_lstm_v2_best.pth'}")
print(f"[SUCCESS] Saved: {MODELS_DIR / 'motion_lstm_v2_final.pth'}")


# ──────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Final Evaluation (Best Model)")
print("=" * 60)

# Load best model
model.load_state_dict(torch.load(MODELS_DIR / "motion_lstm_v2_best.pth", map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

print(f"\n  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-score:  {f1:.4f}")

print("\n" + "—" * 60)
print("Classification Report:")
print("—" * 60)
print(classification_report(all_labels, all_preds, target_names=MOTION_LABELS, digits=4))

print("\n" + "—" * 60)
print("Confusion Matrix:")
print("—" * 60)
cm = confusion_matrix(all_labels, all_preds)
# Print header
header = "       " + " ".join(f"{i:>4}" for i in range(NUM_CLASSES))
print(header)
for i in range(NUM_CLASSES):
    row = f"  {i:>3}: " + " ".join(f"{cm[i, j]:>4}" for j in range(NUM_CLASSES))
    print(row)

# ──────────────────────────────────────────────────────────────────────
# SAVE CONFIGS
# ──────────────────────────────────────────────────────────────────────
model_config = {
    "model_type": "LSTM",
    "version": "v2",
    "input_size": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "num_classes": NUM_CLASSES,
    "dropout": DROPOUT,
    "max_sequence_length": MAX_LEN,
    "velocity_scale": VELOCITY_SCALE,
    "motion_labels": MOTION_LABELS,
    "accuracy": round(accuracy * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "best_epoch": best_epoch,
    "total_epochs": EPOCHS,
}
with open(MODELS_DIR / "model_config_v2.json", "w") as f:
    json.dump(model_config, f, indent=2)
print(f"\n[SUCCESS] Saved: {MODELS_DIR / 'model_config_v2.json'}")

eval_report = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4),
    "confusion_matrix": cm.tolist(),
    "per_class": {}
}
for i, label in enumerate(MOTION_LABELS):
    mask = all_labels == i
    if mask.sum() > 0:
        class_acc = (all_preds[mask] == i).sum() / mask.sum()
        eval_report["per_class"][label] = {
            "accuracy": round(float(class_acc), 4),
            "support": int(mask.sum()),
        }
with open(MODELS_DIR / "evaluation_report_v2.json", "w") as f:
    json.dump(eval_report, f, indent=2)
print(f"[SUCCESS] Saved: {MODELS_DIR / 'evaluation_report_v2.json'}")

# ──────────────────────────────────────────────────────────────────────
# SAVE TRAINING CURVES DATA
# ──────────────────────────────────────────────────────────────────────
training_history = {
    "train_losses": train_losses,
    "val_losses": val_losses,
    "train_accs": train_accs,
    "val_accs": val_accs,
}
with open(MODELS_DIR / "training_history_v2.json", "w") as f:
    json.dump(training_history, f)
print(f"[SUCCESS] Saved: {MODELS_DIR / 'training_history_v2.json'}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
