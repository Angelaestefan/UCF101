#!/usr/bin/env python3
"""
UCF101 Skeleton-Based Action Recognition - Prediction Interface
Console-based prediction interface for CNN+RNN model
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Target classes
TARGET_CLASSES = ['JumpRope', 'JumpingJack', 'PushUps', 'Lunges', 'BodyWeightSquats']


class SkeletonDataset(Dataset):
    """Dataset for skeleton-based action recognition."""
    def __init__(self, pkl_path, split_or_list="train1", frames=32, normalize=True):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        self.frames = frames
        self.normalize = normalize
        
        if isinstance(split_or_list, str):
            split_dict = data['split']
            frame_dirs = set(split_dict[split_or_list])
        else:
            frame_dirs = set(split_or_list)
        
        self.annotations = [a for a in data['annotations'] if a['frame_dir'] in frame_dirs]
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        kp = np.array(ann['keypoint'], dtype=np.float32)  # (T, V, C)
        label = ann['label']
        T, V, C = kp.shape
        
        # Temporal center-crop or pad
        if T >= self.frames:
            start = (T - self.frames) // 2
            kp = kp[start:start + self.frames]
        else:
            pad = np.repeat(kp[-1:], self.frames - T, axis=0)
            kp = np.concatenate([kp, pad], axis=0)
        
        # Normalize
        if self.normalize:
            center_joint = kp[:, 0:1, :]
            kp = kp - center_joint
            kp = kp / 100.0
        
        return torch.tensor(kp, dtype=torch.float32), int(label)


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


def predict_single_sample(model, sample, device):
    """Predict class for a single sample."""
    model.eval()
    with torch.no_grad():
        sample = sample.unsqueeze(0).to(device)  # Add batch dimension
        output = model(sample)
        probabilities = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_class].item()
    
    return pred_class, confidence, probabilities[0].cpu().numpy()


def predict_from_dataset(model, dataset, device, num_samples=10):
    """Predict on multiple samples from dataset and display results."""
    print(f'\n{"=" * 80}')
    print(f'Predicting on {num_samples} random samples from dataset...')
    print('=' * 80)
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    correct = 0
    for i, idx in enumerate(indices):
        sample, true_label = dataset[idx]
        pred_class, confidence, probs = predict_single_sample(model, sample, device)
        
        is_correct = pred_class == true_label
        correct += int(is_correct)
        
        status = "✓" if is_correct else "✗"
        print(f'\nSample {i+1}:')
        print(f'  True Label:      {TARGET_CLASSES[true_label]}')
        print(f'  Predicted:       {TARGET_CLASSES[pred_class]} ({confidence:.2%} confidence) {status}')
        print(f'  All Probabilities:')
        for class_idx, class_name in enumerate(TARGET_CLASSES):
            print(f'    {class_name:20s}: {probs[class_idx]:.2%}')
    
    accuracy = correct / len(indices)
    print(f'\n{"=" * 80}')
    print(f'Accuracy: {correct}/{len(indices)} = {accuracy:.2%}')
    print('=' * 80)


def interactive_mode(model, dataset, device):
    """Interactive prediction mode."""
    print(f'\n{"=" * 80}')
    print('Interactive Prediction Mode')
    print('=' * 80)
    print(f'Dataset has {len(dataset)} samples (indices 0-{len(dataset)-1})')
    print('Enter sample index to predict, or "q" to quit')
    print('=' * 80)
    
    while True:
        user_input = input('\nEnter sample index (or "q" to quit): ').strip()
        
        if user_input.lower() == 'q':
            print('Exiting interactive mode.')
            break
        
        try:
            idx = int(user_input)
            if idx < 0 or idx >= len(dataset):
                print(f'Error: Index must be between 0 and {len(dataset)-1}')
                continue
            
            sample, true_label = dataset[idx]
            pred_class, confidence, probs = predict_single_sample(model, sample, device)
            
            is_correct = pred_class == true_label
            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
            
            print(f'\n{"─" * 80}')
            print(f'Sample {idx}:')
            print(f'  True Label:      {TARGET_CLASSES[true_label]}')
            print(f'  Predicted:       {TARGET_CLASSES[pred_class]} ({confidence:.2%} confidence) {status}')
            print(f'  All Probabilities:')
            for class_idx, class_name in enumerate(TARGET_CLASSES):
                bar_length = int(probs[class_idx] * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                print(f'    {class_name:20s}: {bar} {probs[class_idx]:.2%}')
            print('─' * 80)
            
        except ValueError:
            print('Error: Please enter a valid integer index or "q" to quit')


def main():
    parser = argparse.ArgumentParser(description='UCF101 Skeleton Action Recognition - Prediction Interface')
    parser.add_argument('--model', type=str, default='cnn_rnn_improved.pth', 
                       help='Path to trained model weights')
    parser.add_argument('--data', type=str, default='ucf101_5classes_skeleton.pkl', 
                       help='Path to dataset PKL')
    parser.add_argument('--split', type=str, default='test1', 
                       help='Dataset split to use (train1, test1, etc.)')
    parser.add_argument('--mode', type=str, default='random', choices=['random', 'interactive'],
                       help='Prediction mode: random (predict on random samples) or interactive (enter sample indices)')
    parser.add_argument('--num_samples', type=int, default=10, 
                       help='Number of random samples to predict (only for random mode)')
    
    args = parser.parse_args()
    
    print('=' * 80)
    print('UCF101 Skeleton-Based Action Recognition - Prediction Interface')
    print('=' * 80)
    print(f'Configuration:')
    print(f'  - Model: {args.model}')
    print(f'  - Dataset: {args.data}')
    print(f'  - Split: {args.split}')
    print(f'  - Mode: {args.mode}')
    print(f'  - Device: {device}')
    print('=' * 80)
    
    # Load dataset
    try:
        dataset = SkeletonDataset(args.data, split_or_list=args.split)
        print(f'\nDataset loaded: {len(dataset)} samples in {args.split}')
    except FileNotFoundError:
        print(f'\nError: Dataset file {args.data} not found!')
        return
    except Exception as e:
        print(f'\nError loading dataset: {e}')
        return
    
    # Load model
    try:
        model = CNN_RNN_Improved().to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        print(f'Model loaded successfully from {args.model}')
    except FileNotFoundError:
        print(f'\nError: Model file {args.model} not found!')
        print('Please train the model first using train_cnn_rnn.py or the notebook.')
        return
    except Exception as e:
        print(f'\nError loading model: {e}')
        return
    
    # Prediction mode
    if args.mode == 'random':
        predict_from_dataset(model, dataset, device, num_samples=args.num_samples)
    elif args.mode == 'interactive':
        interactive_mode(model, dataset, device)


if __name__ == '__main__':
    main()
