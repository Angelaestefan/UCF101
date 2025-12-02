#!/usr/bin/env python3
"""
UCF101 Skeleton-Based Action Recognition - Training Script
Implements CNN+RNN architecture with 3-fold cross-validation
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')

# Target classes
TARGET_CLASSES = ['JumpRope', 'JumpingJack', 'PushUps', 'Lunges', 'BodyWeightSquats']


class SkeletonDataset(Dataset):
    """Dataset for skeleton-based action recognition with temporal augmentation support."""
    def __init__(self, pkl_path, split_or_list="train1", frames=32, normalize=True, augment=False):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.frames = frames
        self.normalize = normalize
        self.augment = augment
        
        if isinstance(split_or_list, str):
            split_dict = data['split']
            if split_or_list not in split_dict:
                raise ValueError(f"Split '{split_or_list}' not found in PKL")
            frame_dirs = set(split_dict[split_or_list])
        else:
            frame_dirs = set(split_or_list)
        
        self.annotations = [a for a in data['annotations'] if a['frame_dir'] in frame_dirs]
        if not self.annotations:
            raise ValueError(f"No annotations found for split '{split_or_list}'")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        kp = np.array(ann['keypoint'], dtype=np.float32)  # (T, V, C)
        label = ann['label']
        T, V, C = kp.shape
        
        # Temporal crop or pad (with optional augmentation)
        if T >= self.frames:
            if self.augment:
                max_start = T - self.frames
                start = np.random.randint(0, max_start + 1)
            else:
                start = (T - self.frames) // 2
            kp = kp[start:start + self.frames]
        else:
            pad = np.repeat(kp[-1:], self.frames - T, axis=0)
            kp = np.concatenate([kp, pad], axis=0)
        
        # Normalize spatial coordinates
        if self.normalize:
            center_joint = kp[:, 0:1, :]
            kp = kp - center_joint
            kp = kp / 100.0
        
        return torch.tensor(kp, dtype=torch.float32), int(label)


def split_train_val_by_groups(pkl_path, train_split='train1', val_ratio=0.2, random_seed=42):
    """Split train set into train/val by video groups to avoid leakage."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    train_frame_dirs = set(data['split'][train_split])
    
    # Group by group_id
    group_to_videos = {}
    for ann in data['annotations']:
        if ann['frame_dir'] in train_frame_dirs:
            gid = ann['group_id']
            if gid not in group_to_videos:
                group_to_videos[gid] = []
            group_to_videos[gid].append(ann['frame_dir'])
    
    group_to_videos = {gid: list(set(vids)) for gid, vids in group_to_videos.items()}
    
    # Shuffle groups and split
    groups = list(group_to_videos.keys())
    np.random.seed(random_seed)
    np.random.shuffle(groups)
    
    n_val_groups = max(1, int(len(groups) * val_ratio))
    val_groups = set(groups[:n_val_groups])
    train_groups = set(groups[n_val_groups:])
    
    train_ids = [vid for gid in train_groups for vid in group_to_videos[gid]]
    val_ids = [vid for gid in val_groups for vid in group_to_videos[gid]]
    
    print(f'Train groups: {len(train_groups)}, Val groups: {len(val_groups)}')
    print(f'Train samples: {len(train_ids)}, Val samples: {len(val_ids)}')
    
    return train_ids, val_ids


class CNN_RNN_Baseline(nn.Module):
    """Baseline CNN+RNN model for skeleton action recognition."""
    def __init__(self, num_classes=5, input_frames=32, num_joints=17, num_coords=2, 
                 cnn_channels=64, lstm_hidden=128):
        super().__init__()
        self.num_frames = input_frames
        self.num_joints = num_joints
        self.num_coords = num_coords
        
        # Spatial CNN
        self.conv1 = nn.Conv2d(1, cnn_channels, kernel_size=(3, 1), padding=(1, 0))
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels*2, kernel_size=(3, 1), padding=(1, 0))
        self.pool = nn.AdaptiveAvgPool2d((1, num_coords))
        
        # Temporal LSTM
        self.lstm = nn.LSTM(cnn_channels*2 * num_coords, lstm_hidden, batch_first=True)
        
        # Classifier
        self.fc = nn.Linear(lstm_hidden, num_classes)
    
    def forward(self, x):
        batch_size, T, V, C = x.shape
        
        # Spatial CNN
        x = x.view(batch_size * T, 1, V, C)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(batch_size, T, -1)
        
        # Temporal LSTM
        _, (h_n, _) = self.lstm(x)
        x = h_n.squeeze(0)
        
        # Classifier
        x = self.fc(x)
        return x


class CNN_RNN_Improved(nn.Module):
    """Improved CNN+RNN with BatchNorm, Dropout, and Bidirectional LSTM."""
    def __init__(self, num_classes=5, input_frames=32, num_joints=17, num_coords=2, 
                 cnn_channels=64, lstm_hidden=128, dropout=0.3):
        super().__init__()
        self.num_frames = input_frames
        self.num_joints = num_joints
        self.num_coords = num_coords
        
        # Spatial CNN with BatchNorm
        self.conv1 = nn.Conv2d(1, cnn_channels, kernel_size=(3, 1), padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(cnn_channels)
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels*2, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(cnn_channels*2)
        self.pool = nn.AdaptiveAvgPool2d((1, num_coords))
        
        # Temporal Bidirectional LSTM with Dropout
        self.lstm = nn.LSTM(cnn_channels*2 * num_coords, lstm_hidden, 
                           batch_first=True, bidirectional=True, dropout=dropout)
        
        # Classifier with Dropout
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
    
    def forward(self, x):
        batch_size, T, V, C = x.shape
        
        # Spatial CNN
        x = x.view(batch_size * T, 1, V, C)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(batch_size, T, -1)
        
        # Temporal LSTM
        _, (h_n, _) = self.lstm(x)
        x = torch.cat([h_n[0], h_n[1]], dim=1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                epochs, device, model_name='model'):
    """Train model and return history."""
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'{model_name} | Epoch {epoch+1}/{epochs} | '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return history


def main():
    """Main training script with 3-fold cross-validation."""
    
    # Configuration
    PKL_PATH = 'ucf101_5classes_skeleton.pkl'
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 0.001
    
    print('=' * 80)
    print('UCF101 Skeleton-Based Action Recognition - CNN+RNN Training')
    print('=' * 80)
    print(f'Configuration:')
    print(f'  - Dataset: {PKL_PATH}')
    print(f'  - Batch Size: {BATCH_SIZE}')
    print(f'  - Epochs: {EPOCHS}')
    print(f'  - Learning Rate: {LR}')
    print(f'  - Device: {device}')
    print('=' * 80)
    
    # Check if PKL exists
    try:
        with open(PKL_PATH, 'rb') as f:
            data = pickle.load(f)
        print(f'\nDataset loaded: {len(data["annotations"])} samples')
    except FileNotFoundError:
        print(f'\nError: {PKL_PATH} not found!')
        print('Please run the data filtering section in the notebook first.')
        return
    
    criterion = nn.CrossEntropyLoss()
    
    # 3-Fold Evaluation
    results_3splits = []
    
    for split_idx in range(1, 4):
        print(f'\n\n{"=" * 80}')
        print(f'Split {split_idx}/3')
        print('=' * 80)
        train_split_name = f'train{split_idx}'
        test_split_name = f'test{split_idx}'
        
        # Group-aware split
        train_ids, val_ids = split_train_val_by_groups(PKL_PATH, train_split=train_split_name, val_ratio=0.2)
        
        # Datasets
        train_ds = SkeletonDataset(PKL_PATH, split_or_list=train_ids, augment=True)
        val_ds = SkeletonDataset(PKL_PATH, split_or_list=val_ids, augment=False)
        test_ds = SkeletonDataset(PKL_PATH, split_or_list=test_split_name, augment=False)
        
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_ld = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
        
        # Train improved model
        print(f'\nTraining CNN+RNN Improved on Split {split_idx}...')
        model_split = CNN_RNN_Improved().to(device)
        optimizer_split = optim.Adam(model_split.parameters(), lr=LR)
        
        history_split = train_model(
            model_split, train_ld, val_ld, criterion, optimizer_split, 
            EPOCHS, device, model_name=f'Split{split_idx}'
        )
        
        # Test
        _, test_acc_split = eval_epoch(model_split, test_ld, criterion, device)
        print(f'\nSplit {split_idx} Test Accuracy: {test_acc_split:.4f}')
        
        results_3splits.append({
            'split': split_idx,
            'test_acc': test_acc_split,
            'history': history_split
        })
    
    # Summary statistics
    test_accs = [r['test_acc'] for r in results_3splits]
    print('\n\n' + '=' * 80)
    print('3-Fold Evaluation Summary')
    print('=' * 80)
    print(f'Test Accuracies: {[f"{acc:.4f}" for acc in test_accs]}')
    print(f'Mean: {np.mean(test_accs):.4f}')
    print(f'Std: {np.std(test_accs):.4f}')
    print('=' * 80)
    
    # Save results
    results_summary = {
        '3splits': results_3splits,
        'mean_test_acc': np.mean(test_accs),
        'std_test_acc': np.std(test_accs)
    }
    
    with open('results_cnn_rnn_3splits.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f'\nResults saved to: results_cnn_rnn_3splits.pkl')
    
    # Plot 3-fold results
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(test_accs))
    ax.bar(x_pos, test_accs, align='center', alpha=0.7, color='skyblue', edgecolor='navy')
    ax.axhline(y=np.mean(test_accs), color='red', linestyle='--', label=f'Mean: {np.mean(test_accs):.4f}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'Split {i+1}' for i in range(len(test_accs))])
    ax.set_ylabel('Test Accuracy')
    ax.set_title('CNN+RNN Improved - 3-Fold Test Accuracy')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('3fold_test_accuracy.png', dpi=150)
    print('Plot saved to: 3fold_test_accuracy.png')


if __name__ == '__main__':
    main()
