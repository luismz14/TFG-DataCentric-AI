"""
ModelTrain.py

Training module for colorectal polyp classification.
Implements a lightweight Model-Centric baseline (EfficientNet-B0)
"""

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

from .PolypClassifier import PolypClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'



def perform_clinical_data_split(metadata_path: str | Path, train_ratio: float = 0.8, random_state: int = 42):
    """
    Performs a strict group-based split to prevent Data Leakage.

    A standard random split would distribute images of the exact same physical polyp into both Train and Validation sets.
    This split strategy split by unique polyp identifiers.
    """
    df = pd.read_csv(metadata_path)
    # Some annotations of histology are missing.
    df = df.dropna(subset=['histology'])
    
    unique_polyp_identifiers = ['patient_id', 'day', 'R', 'F']
    df['Group_ID'] = df[unique_polyp_identifiers].astype(str).apply('_'.join, axis=1)
    
    X = df['filename']
    y = df['histology'] 
    groups = df['Group_ID']
    
    # GroupShuffleSplit ensures all records sharing the same Group_ID end up in the same split
    gss = GroupShuffleSplit(n_splits=1, test_size=(1.0 - train_ratio), random_state=random_state)
    train_indices, val_indices = next(gss.split(X, y, groups))
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    return train_df, val_df


class PolypDataset(Dataset):
    """
    Lazy-loading custom Dataset.
    """
    def __init__(self, dataframe, images_dir: str | Path, transform=None):
        self.df = dataframe
        self.images_dir = Path(images_dir)
        self.transform = transform
        
        # Maps histologies to integers for PyTorch tensor compatibility
        self.label_map = {
            'Adenoma': 0, 
            'Sessile_serrated_adenoma': 1, 
            'Hyperplastic': 2, 
            'Adenocarcinoma': 3
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = self.images_dir / row['filename']
        label_idx = self.label_map[row['histology']]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Corrupted or missing image at: {img_path}")
            
        # OpenCV uses BGR. EfficientNet-B0 backbone expects RGB (ImageNet standard).
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label_idx


def evaluate_model(model, val_loader, criterion, device):
    """
    Computes validation metrics without updating model weights.
    """
    model.eval()
    
    val_loss = 0.0
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted_classes = torch.max(outputs, 1)
            
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    
    macro_f1 = f1_score(all_true_labels, all_predictions, average='macro')
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    return avg_loss, macro_f1, cm


def plot_and_save_metrics(train_losses, val_losses, val_f1_scores, best_cm, class_names, save_dir=None):
    """
    Generates experimental artifacts for the TFG memory and methodological evaluation.
    """
    if save_dir is None:
        save_dir_path = RESULTS_DIR
    else:        
        save_dir_path = RESULTS_DIR / Path(save_dir)
    
    save_dir_path.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    # Loss Evolution Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.title('Loss Evolution across Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('CrossEntropy Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir_path / 'loss_evolution.png')
    plt.close()

    # F1-Score Evolution Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1_scores, label='Validation F1-Score (Macro)', color='green')
    plt.title('Validation F1-Score Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir_path / 'f1_score_evolution.png')
    plt.close()

    # Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Best Model)')
    plt.ylabel('True Clinical Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir_path / 'confusion_matrix.png')
    plt.close()
    
    print(f"Artifacts successfully written to '{save_dir_path}'.")


def train(csv_name: str | Path, images_dir_name: str | Path, save_dir: str):
    """
    Orchestrates the data ingestion, sampling logic, optimization loop, 
    and checkpointing mechanism.
    """
    # 1. Data Ingestion & Split
    csv_path = DATA_DIR / Path(csv_name)
    train_metadata_df, val_metadata_df = perform_clinical_data_split(csv_path)

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), # Mandatory dimension for EfficientNet-B0
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    images_dir = DATA_DIR / Path(images_dir_name)
    train_dataset = PolypDataset(train_metadata_df, images_dir=images_dir, transform=data_transforms)
    val_dataset = PolypDataset(val_metadata_df, images_dir=images_dir, transform=data_transforms)

    # 2. Resampling Strategy
    class_counts = train_metadata_df['histology'].value_counts().to_dict()
    sample_weights = [1.0 / class_counts[histology] for histology in train_metadata_df['histology']]

    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. Model & Optimizer Instantiation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware assigned for tensor computations: {device}")
    
    model = PolypClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100

    # Telemetry trackers
    history_train_loss = []
    history_val_loss = []
    history_val_f1 = []
    
    best_f1_score = 0.0
    best_confusion_matrix = None
    
    experiment_dir = RESULTS_DIR / save_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)
    best_model_weights_path = experiment_dir / 'best_baseline_model.pth'

    # 4. Training Loop
    print(f"Initiating Baseline Training Phase. Saving results to: {experiment_dir}")
    pbar = tqdm(range(num_epochs), desc="Training Progress", unit="epoch")
    
    for epoch in pbar:
        
        # --- TRAINING PASS ---
        model.train()
        running_train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        epoch_train_loss = running_train_loss / len(train_loader)
        
        # --- VALIDATION PASS ---
        epoch_val_loss, epoch_val_f1, current_cm = evaluate_model(model, val_loader, criterion, device)
        
        history_train_loss.append(epoch_train_loss)
        history_val_loss.append(epoch_val_loss)
        history_val_f1.append(epoch_val_f1)
        
        # --- CHECKPOINTING MECHANISM ---
        checkpoint_status = ""
        if epoch_val_f1 > best_f1_score:
            best_f1_score = epoch_val_f1
            best_confusion_matrix = current_cm
            torch.save(model.state_dict(), best_model_weights_path)
            checkpoint_status = " 💾 [SAVED]"
            
        pbar.set_postfix({
            'Train Loss': f"{epoch_train_loss:.4f}",
            'Val Loss': f"{epoch_val_loss:.4f}",
            'Val F1': f"{epoch_val_f1:.4f}{checkpoint_status}"
        })
    
    print()
    print(f"Optimization sequence completed. Optimal Validation F1-Score: {best_f1_score:.4f}")

    # Generate visual artifacts based on the final telemetry
    class_names = ['Adenoma', 'Sessile Serrated', 'Hyperplastic', 'Adenocarcinoma']
    plot_and_save_metrics(history_train_loss, history_val_loss, history_val_f1, best_confusion_matrix, class_names, save_dir=save_dir)
    
    return model, val_loader