"""Task definition for the Private Shapley example."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import cast, Tuple, List, Dict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torchvision.transforms import Compose, Normalize, ToTensor
from collections import OrderedDict

class Net(nn.Module):
    """Simple CNN for CIFAR-10."""
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def get_weights(net: nn.Module) -> List:
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net: nn.Module, parameters: List) -> None:
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

# Cache for FederatedDataset
fds = None

def load_data(partition_id: int, num_partitions: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 data for a specific partition.
    
    Args:
        partition_id: ID of the partition to load
        num_partitions: Total number of partitions
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    global fds
    
    # Initialize FederatedDataset once
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    
    # Load partition
    partition = fds.load_partition(partition_id)
    
    # Apply transformations
    def apply_transforms(batch):
        """Apply transforms to the partition data."""
        transforms = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        batch["img"] = [transforms(img) for img in batch["img"]]
        return batch
    
    # Set format and apply transforms
    partition.set_format("numpy")
    partition_split = partition.train_test_split(test_size=0.2, seed=42)
    partition_split = partition_split.with_transform(apply_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        cast(Dataset, partition_split["train"]),
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        cast(Dataset, partition_split["test"]),
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

def train(
    net: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device
) -> Dict:
    """Train the model on the training set.
    
    Args:
        net: The neural network model
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to train on (CPU/GPU)
        
    Returns:
        Dictionary of training metrics
    """
    # Move model to device
    net.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    # Training loop
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Get the inputs and labels
            inputs = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Track statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - "
              f"Accuracy: {100 * correct / total:.2f}%")
    
    return {
        "train_loss": running_loss / len(train_loader),
        "train_accuracy": correct / total
    }

def test(
    net: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate the model on the test set.
    
    Args:
        net: The neural network model
        test_loader: DataLoader for test data
        device: Device to evaluate on (CPU/GPU)
        
    Returns:
        Tuple of (loss, accuracy)
    """
    # Move model to device
    net.to(device)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation mode
    net.eval()
    correct = 0
    total = 0
    loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate average loss and accuracy
    avg_loss = loss / len(test_loader)
    accuracy = correct / total
    
    print(f"Test - Loss: {avg_loss:.4f} - Accuracy: {100 * accuracy:.2f}%")
    
    return avg_loss, accuracy