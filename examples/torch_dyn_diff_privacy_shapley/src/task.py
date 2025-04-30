from collections import OrderedDict
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

fds = None  # Cache FederatedDataset


class Net(nn.Module):
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


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_data(partition_id: int, num_partitions: int, batch_size: int, alpha: float):
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, alpha=alpha, partition_by="label"
        )
        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": partitioner},
        )

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    def apply_transforms(batch):
        batch["image"] = torch.stack(
            [pytorch_transforms(img) for img in batch["image"]]
        )
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    train_loader = DataLoader(
        cast(Dataset, partition_train_test["train"]),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        cast(Dataset, partition_train_test["test"]),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def train(net, train_loader, privacy_engine, optimizer, target_delta, device, epochs=1):
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    for _ in range(epochs):
        for batch in train_loader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    return epsilon


def test(net, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.to(device)
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
