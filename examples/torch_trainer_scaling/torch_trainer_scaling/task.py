from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.datasets import MNIST, FakeData
from torchvision.transforms import Compose, Normalize, ToTensor

DatasetName = Literal["mnist", "fake"]
TaskProfile = Literal["repo", "flower_fl_dp_sa"]


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


class FlowerTutorialNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def build_model(task_profile: TaskProfile = "repo") -> nn.Module:
    if task_profile == "flower_fl_dp_sa":
        return FlowerTutorialNet()
    return Net()


def get_weights(net: nn.Module) -> list:
    return [value.cpu().numpy() for _, value in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: list) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict((k, torch.tensor(v)) for k, v in params_dict)
    net.load_state_dict(state_dict, strict=True)


@dataclass
class DatasetBundle:
    train_partitions: list[Dataset]
    test_dataset: Dataset


_DATASET_CACHE: dict[
    tuple[DatasetName, TaskProfile, str, int, int | None, int | None, int], DatasetBundle
] = {}


def _make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _build_fake_datasets(
    num_trainers: int,
    max_train_samples_per_trainer: int | None,
    max_test_samples: int | None,
    seed: int,
) -> tuple[Dataset, Dataset]:
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    per_trainer = max_train_samples_per_trainer or 256
    total_train_samples = per_trainer * num_trainers
    test_samples = max_test_samples or 1024
    train_dataset = FakeData(
        size=total_train_samples,
        image_size=(1, 28, 28),
        num_classes=10,
        transform=transform,
        random_offset=seed,
    )
    test_dataset = FakeData(
        size=test_samples,
        image_size=(1, 28, 28),
        num_classes=10,
        transform=transform,
        random_offset=seed + 1,
    )
    return train_dataset, test_dataset


def _build_mnist_datasets(
    data_dir: str,
    max_test_samples: int | None,
    seed: int,
) -> tuple[Dataset, Dataset]:
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset: Dataset = MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    if max_test_samples is not None and max_test_samples < len(test_dataset):
        indices = torch.randperm(
            len(test_dataset), generator=_make_generator(seed + 1)
        ).tolist()[:max_test_samples]
        test_dataset = Subset(test_dataset, indices)

    return train_dataset, test_dataset


def _subsample_dataset(dataset: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    indices = torch.randperm(len(dataset), generator=_make_generator(seed)).tolist()[:max_samples]
    return Subset(dataset, indices)


def _build_mnist_datasets_flower_tutorial(
    data_dir: str,
    num_trainers: int,
    max_train_samples_per_trainer: int | None,
    max_test_samples: int | None,
    seed: int,
) -> DatasetBundle:
    del data_dir
    partitioner = IidPartitioner(num_partitions=num_trainers)
    federated_dataset = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": partitioner},
    )
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    def apply_transforms(batch):
        batch["image"] = [transform(image) for image in batch["image"]]
        return batch

    train_partitions: list[Dataset] = []
    test_partitions: list[Dataset] = []
    for partition_id in range(num_trainers):
        partition = federated_dataset.load_partition(partition_id)
        split = partition.train_test_split(test_size=0.2, seed=42).with_transform(
            apply_transforms
        )
        train_dataset = _subsample_dataset(
            split["train"],
            max_train_samples_per_trainer,
            seed + partition_id,
        )
        train_partitions.append(train_dataset)
        test_partitions.append(split["test"])

    test_dataset: Dataset = ConcatDataset(test_partitions)
    test_dataset = _subsample_dataset(test_dataset, max_test_samples, seed + 1)
    return DatasetBundle(train_partitions=train_partitions, test_dataset=test_dataset)


def _partition_indices(
    dataset_size: int,
    num_trainers: int,
    max_train_samples_per_trainer: int | None,
    seed: int,
) -> list[list[int]]:
    indices = torch.randperm(dataset_size, generator=_make_generator(seed)).tolist()

    if max_train_samples_per_trainer is not None:
        required = num_trainers * max_train_samples_per_trainer
        if required > dataset_size:
            raise ValueError(
                "Requested more train samples than the dataset contains. "
                f"Needed {required}, available {dataset_size}."
            )
        selected = indices[:required]
        return [
            selected[
                i * max_train_samples_per_trainer : (i + 1)
                * max_train_samples_per_trainer
            ]
            for i in range(num_trainers)
        ]

    base, remainder = divmod(dataset_size, num_trainers)
    partitions: list[list[int]] = []
    start = 0
    for trainer_idx in range(num_trainers):
        size = base + (1 if trainer_idx < remainder else 0)
        partitions.append(indices[start : start + size])
        start += size
    return partitions


def load_datasets(
    dataset_name: DatasetName,
    data_dir: str,
    num_trainers: int,
    max_train_samples_per_trainer: int | None,
    max_test_samples: int | None,
    seed: int,
    task_profile: TaskProfile = "repo",
) -> DatasetBundle:
    cache_key = (
        dataset_name,
        task_profile,
        data_dir,
        num_trainers,
        max_train_samples_per_trainer,
        max_test_samples,
        seed,
    )
    if cache_key in _DATASET_CACHE:
        return _DATASET_CACHE[cache_key]

    if dataset_name == "fake":
        train_dataset, test_dataset = _build_fake_datasets(
            num_trainers=num_trainers,
            max_train_samples_per_trainer=max_train_samples_per_trainer,
            max_test_samples=max_test_samples,
            seed=seed,
        )
        partition_indices = _partition_indices(
            dataset_size=len(train_dataset),
            num_trainers=num_trainers,
            max_train_samples_per_trainer=max_train_samples_per_trainer,
            seed=seed,
        )
        train_partitions = [Subset(train_dataset, indices) for indices in partition_indices]
        bundle = DatasetBundle(train_partitions=train_partitions, test_dataset=test_dataset)
        _DATASET_CACHE[cache_key] = bundle
        return bundle
    if task_profile == "flower_fl_dp_sa":
        bundle = _build_mnist_datasets_flower_tutorial(
            data_dir=data_dir,
            num_trainers=num_trainers,
            max_train_samples_per_trainer=max_train_samples_per_trainer,
            max_test_samples=max_test_samples,
            seed=seed,
        )
        _DATASET_CACHE[cache_key] = bundle
        return bundle
    else:
        train_dataset, test_dataset = _build_mnist_datasets(
            data_dir=data_dir,
            max_test_samples=max_test_samples,
            seed=seed,
        )

    partition_indices = _partition_indices(
        dataset_size=len(train_dataset),
        num_trainers=num_trainers,
        max_train_samples_per_trainer=max_train_samples_per_trainer,
        seed=seed,
    )
    train_partitions = [Subset(train_dataset, indices) for indices in partition_indices]
    bundle = DatasetBundle(train_partitions=train_partitions, test_dataset=test_dataset)
    _DATASET_CACHE[cache_key] = bundle
    return bundle


def get_trainloader(
    partition_id: int,
    batch_size: int,
    dataset_name: DatasetName,
    data_dir: str,
    num_trainers: int,
    max_train_samples_per_trainer: int | None,
    max_test_samples: int | None,
    seed: int,
    num_workers: int = 0,
    persistent_workers: bool = False,
    task_profile: TaskProfile = "repo",
) -> DataLoader:
    bundle = load_datasets(
        dataset_name=dataset_name,
        task_profile=task_profile,
        data_dir=data_dir,
        num_trainers=num_trainers,
        max_train_samples_per_trainer=max_train_samples_per_trainer,
        max_test_samples=max_test_samples,
        seed=seed,
    )
    return DataLoader(
        bundle.train_partitions[partition_id],
        batch_size=batch_size,
        shuffle=True,
        generator=_make_generator(seed + partition_id),
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def get_testloader(
    batch_size: int,
    dataset_name: DatasetName,
    data_dir: str,
    num_trainers: int,
    max_train_samples_per_trainer: int | None,
    max_test_samples: int | None,
    seed: int,
    num_workers: int = 0,
    persistent_workers: bool = False,
    task_profile: TaskProfile = "repo",
) -> DataLoader:
    bundle = load_datasets(
        dataset_name=dataset_name,
        task_profile=task_profile,
        data_dir=data_dir,
        num_trainers=num_trainers,
        max_train_samples_per_trainer=max_train_samples_per_trainer,
        max_test_samples=max_test_samples,
        seed=seed,
    )
    return DataLoader(
        bundle.test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers and num_workers > 0,
    )


def _unpack_batch(batch):
    if isinstance(batch, dict):
        return batch["image"], batch["label"]
    return batch


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> float:
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()

    total_loss = 0.0
    total_batches = 0

    for _ in range(epochs):
        for batch in trainloader:
            images, labels = _unpack_batch(batch)
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    net.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in testloader:
            images, labels = _unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            total_loss += float(criterion(outputs, labels).item())
            total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total_examples += int(labels.size(0))

    average_loss = total_loss / max(len(testloader), 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy
