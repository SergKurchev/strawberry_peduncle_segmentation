"""
Training script for AffinityNet.
Can be run standalone or imported into notebook.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def train_affinity_net(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    """
    Train the Affinity Network.
    
    Args:
        model: AffinityNet instance
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        epochs: Number of training epochs
        lr: Learning rate
        device: 'cuda' or 'cpu'
    
    Returns:
        history: Dict with training history
    """
    model = model.to(device)
    
    # Loss & Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        # === TRAINING ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [TRAIN]')
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # Forward
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # For precision/recall
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [VAL]')
            for features, labels in pbar:
                features = features.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Accuracy
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # Precision/Recall metrics
                true_positives += ((predicted == 1) & (labels == 1)).sum().item()
                false_positives += ((predicted == 1) & (labels == 0)).sum().item()
                false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
                
                pbar.set_postfix({'loss': loss.item()})
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        # Logging
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        # LR scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_affinity_net.pth')
            print(f'  ✅ Best model saved (acc={val_acc:.4f})')
    
    return history


def plot_training_history(history, save_path='affinity_training.png'):
    """Plot training curves."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['val_acc'], label='Accuracy', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision & Recall
    axes[1, 0].plot(history['val_precision'], label='Precision', color='blue')
    axes[1, 0].plot(history['val_recall'], label='Recall', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1 Score
    axes[1, 1].plot(history['val_f1'], label='F1 Score', color='purple')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Training curves saved to {save_path}")
    
    return fig


if __name__ == '__main__':
    import os
    import json
    from affinity_net import AffinityNet
    from affinity_dataset import AffinityDataset
    
    # Configuration
    DATASET_PATH = 'strawberry_peduncle_segmentation/dataset'
    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # === LOAD DATA ===
    print("\n📂 Loading dataset...")
    with open(os.path.join(DATASET_PATH, 'annotations.json'), 'r') as f:
        coco_data = json.load(f)
    
    train_dataset = AffinityDataset(coco_data, DATASET_PATH, split='train')
    val_dataset = AffinityDataset(coco_data, DATASET_PATH, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)
    
    # === CREATE MODEL ===
    print("\n🧠 Creating AffinityNet...")
    model = AffinityNet(spatial_dim=5, hidden_dims=[32, 16])
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # === TRAIN ===
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    history = train_affinity_net(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        device=device
    )
    
    # === PLOT RESULTS ===
    print("\n📊 Generating plots...")
    plot_training_history(history)
    
    print("\n✅ Training complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
