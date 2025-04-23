from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


# Define the neural network model suitable for MNIST
class Net(nn.Module):
    """Simple CNN Model for MNIST."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # MNIST has 1 channel
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjusted for MNIST image size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = F.relu(self.fc1(x))  # [batch, 128]
        x = self.fc2(x)  # [batch, 10]
        return x


# Utility functions to get and set model weights
def get_weights(net):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# Global FederatedDataset cache
fds = None


def load_data(
    partition_id: int, num_partitions: int, batch_size: int, alpha: float = 0.5
):
    """
    Load MNIST data partition with Non-IID distribution using DirichletPartitioner.
    """
    # alpha α=0.5	Moderate Non-IID (Some label imbalance)
    # α=0.1	Highly Non-IID (Extreme label skew)
    # α (e.g., 10.0) → More IID (
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            alpha=alpha,
            partition_by="label",  # Specifies that partitioning is based on labels
        )
        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Split into train and test (80% train, 20% test)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Define transformations
    pytorch_transforms = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    def apply_transforms(batch):
        """Apply transformations to the data."""
        # Change 'img' to 'image' to match MNIST dataset keys
        batch["image"] = torch.stack(
            [pytorch_transforms(img) for img in batch["image"]]
        )
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(
        partition_train_test["test"], batch_size=batch_size, shuffle=False
    )
    return trainloader, testloader


def train(net, trainloader, valloader, epochs, learning_rate, device):
    """
    Train the model on the training set with gradient clipping.
    """
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()
    val_loss, val_acc = test(net, valloader, device)
    results = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader, device):
    """
    Evaluate the model on the test set.
    """
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = net(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    loss = loss / len(testloader)
    return loss, accuracy
