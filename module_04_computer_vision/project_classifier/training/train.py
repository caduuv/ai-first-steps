"""Training pipeline for CIFAR-10 classifier."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from data.dataset import get_dataloaders, CLASS_NAMES
from models.cnn import CIFAR10CNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
    
    return total_loss / len(loader), correct / total


def main():
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 50)
    print("  CIFAR-10 Classification Training")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    train_loader, test_loader = get_dataloaders(
        batch_size=config['training']['batch_size'],
        augment=config['data']['augment']
    )
    
    # Model
    model = CIFAR10CNN(
        num_classes=10,
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Train
    best_acc = 0
    for epoch in range(config['training']['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1:>2}/{config['training']['epochs']}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
              f"test_loss={test_loss:.4f}, test_acc={test_acc:.2%}")
    
    print(f"\nBest test accuracy: {best_acc:.2%}")
    print("Model saved to best_model.pth")


if __name__ == '__main__':
    main()
