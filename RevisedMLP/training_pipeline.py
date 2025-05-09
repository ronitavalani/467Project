import os
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from enhanced_model import GenreNetEnhanced
from feature_engineering import apply_feature_engineering_pipeline
from eval import evaluate_model_detailed, analyze_misclassifications, visualize_learned_features

# Load data
url = 'https://raw.githubusercontent.com/ronitavalani/467Project/main/spotify_songs.csv'
df = pd.read_csv(url)
df['playlist_genre'] = df['playlist_genre'].astype(str).apply(lambda x: x.split(',')[0].strip())

# Feature engineering
X, y = apply_feature_engineering_pipeline(df, target_col='playlist_genre', use_pca=False)
feature_names = list(X.columns)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Filter low-frequency classes
value_counts = pd.Series(y_encoded).value_counts()
valid_classes = value_counts[value_counts > 1].index
valid_mask = pd.Series(y_encoded).isin(valid_classes)
X = X[valid_mask]
y = y[valid_mask].reset_index(drop=True)
y_encoded = le.fit_transform(y)

# Scale features to zero mean and unit variance
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Custom Train/Dev/Test Split: 70/10/20 - Stratified
X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Dataset class
class AugmentedSongDataset(Dataset):
    def __init__(self, features, labels, augment=False, noise_level=0.05):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
        self.augment = augment
        self.noise_level = noise_level
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.augment and torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
            scale_factor = 1.0 + (torch.rand(x.shape) * 0.2 - 0.1)
            x = x * scale_factor
        return x, y

train_loader = DataLoader(AugmentedSongDataset(X_train, y_train, augment=True), batch_size=64, shuffle=True)
val_loader = DataLoader(AugmentedSongDataset(X_val, y_val, augment=False), batch_size=64)
test_loader = DataLoader(AugmentedSongDataset(X_test, y_test, augment=False), batch_size=64)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

input_dim = X_train.shape[1]
hidden_dims = [512, 256, 128, 64]
output_dim = len(np.unique(y_encoded))
dropout_rate = 0.4
learning_rate = 0.001
weight_decay = 1e-4

model = GenreNetEnhanced(input_dim, hidden_dims, output_dim, dropout_rate)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50, patience=10, device="cpu"):
    model = model.to(device)
    best_val_acc = 0
    patience_counter = 0
    train_losses, val_accuracies = [], []
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            if device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_acc = evaluate(model, val_loader, device)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"New best model saved with accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    return best_val_acc

def evaluate(model, dataloader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
best_val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device=device)
model.load_state_dict(torch.load("best_model.pt"))
model = model.to(device)

test_accuracy = evaluate(model, test_loader, device)
print(f"Test Set Accuracy: {test_accuracy * 100:.2f}%")

eval_results = evaluate_model_detailed(model, test_loader, device, le.classes_)
misclass_df = analyze_misclassifications(model, test_loader, device, le.classes_, feature_names)
visualize_learned_features(model, test_loader, device, le.classes_, n_samples=200)
