import argparse
import warnings
from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler



# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz' for regression)"""

    

    
    def __init__(self, num_features: int) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * (num_features // 8), 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

        
def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = nn.MSELoss()  # Use Mean Squared Error loss for regression
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for _ in range(epochs):
        # for batch in tqdm(trainloader, desc="Training"):
        for inputs, labels in tqdm(trainloader, desc="training"):
            # images = batch["img"]
            # labels = batch["label"]
            
            # inputs = batch.drop(columns='median_house_value')
            # labels = batch['median_house_value']
            
            optimizer.zero_grad()
            # outputs = net(images.to(DEVICE))
            outputs = net(inputs.to(DEVICE))
            loss = criterion(outputs, labels.unsqueeze(1).to(DEVICE))
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = nn.MSELoss()
    loss = 0.0
    r2 = 0.0
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for inputs, labels in tqdm(testloader, "Testing"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            r2 += r2_score(labels.cpu().numpy(), outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
            
    avg_loss = loss / len(testloader.dataset)
    # avg_r2 = r2 / len(testloader.dataset)
    avg_r2 = r2
    mse = mean_squared_error(all_labels, all_outputs)/len(testloader.dataset)
    print('test on the local dataset')
    print('test dataset')
    print(testloader.dataset)
    print("avg_r2 = ", avg_r2)
    return mse, avg_r2

# def train(net, trainloader, epochs):
#     """Train the model on the training set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#     for _ in range(epochs):
#         for batch in tqdm(trainloader, "Training"):
#             images = batch["img"]
#             labels = batch["label"]
#             optimizer.zero_grad()
#             criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
#             optimizer.step()


# def test(net, testloader):
#     """Validate the model on the test set."""
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in tqdm(testloader, "Testing"):
#             images = batch["img"].to(DEVICE)
#             labels = batch["label"].to(DEVICE)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     return loss, accuracy


def load_data(partition_id):
    """Load partition data."""
    housing_dataset = pd.read_csv(f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{partition_id}.csv')
    
    print(f'client number {partition_id} holds the data of {housing_dataset["City"].iloc[0]}')
    print()
    housing_dataset = housing_dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])
    housing_dataset = housing_dataset.dropna()
    # Prepare features and target
    X = housing_dataset.drop(columns='Price').values
    y = housing_dataset['Price'].values

    # Reshape features for CNN input
    num_samples, num_features = X.shape
    X_reshaped = X.reshape((num_samples, 1, num_features))  # Reshape to 3D array for Conv1d (1 channel)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    # min_max_scaler = MinMaxScaler()
    # X_train = min_max_scaler.fit_transform(X_train)
    # X_test = min_max_scaler.transform(X_test)
    # Convert to PyTorch tensors
    
    num_samples_train, _, num_features_train = X_train.shape
    num_samples_test, _, num_features_test = X_test.shape
    X_train = X_train.reshape((num_samples_train, num_features_train))
    X_test = X_test.reshape((num_samples_test, num_features_test))

    # Normalize features
    min_max_scaler = MinMaxScaler()
    X_train_normalized = min_max_scaler.fit_transform(X_train)
    X_test_normalized = min_max_scaler.transform(X_test)

    # Reshape normalized features back to 3D arrays
    X_train_normalized = X_train_normalized.reshape((num_samples_train, 1, num_features_train))
    X_test_normalized = X_test_normalized.reshape((num_samples_test, 1, num_features_test))

    
    # X_train = torch.tensor(X_train, dtype=torch.float32)
    # X_test = torch.tensor(X_test, dtype=torch.float32)
    # y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    # y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    X_train = torch.tensor(X_train_normalized, dtype=torch.float32)
    X_test = torch.tensor(X_test_normalized, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    default=0,
    type=int,
    help="Partition of the dataset divided into 2 iid partitions created artificially.",
)
partition_id = parser.parse_known_args()[0].partition_id

# partition_id = 0

# Load model and data (simple CNN, CIFAR-10)
net = Net(12).to(DEVICE)
trainloader, testloader = load_data(partition_id=partition_id)

# train
train(net, trainloader, 10)
loss, acc = test(net, testloader)

print('loss: ', loss)
# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
