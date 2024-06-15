import argparse
import warnings
from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train(net, trainloader, epochs):
    criterion = nn.MSELoss()  # Use Mean Squared Error loss for regression
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for _ in range(epochs):
        for inputs, labels in tqdm(trainloader, desc="training"):
            optimizer.zero_grad()
            outputs = net(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()

def test(net, testloader):
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
    avg_r2 = r2 / len(testloader.dataset)
    mse = mean_squared_error(all_labels, all_outputs) / len(testloader.dataset)
    print('test on the local dataset')
    print('test dataset')
    print(testloader.dataset)
    print("avg_r2 = ", avg_r2)
    return mse, avg_r2

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

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    # min_max_scaler = MinMaxScaler()
    # X_train = min_max_scaler.fit_transform(X_train)
    # X_test = min_max_scaler.transform(X_test)

    # Convert to PyTorch tensors and reshape for LSTM
    X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, X_train.shape[1])
    X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, X_test.shape[1])
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
    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    default=0,
    type=int,
    help="Partition of the dataset divided into 2 iid partitions created artificially.",
)
partition_id = parser.parse_known_args()[0].partition_id

# Load model and data
input_size = 10  # This should match the number of features
hidden_size = 64
num_layers = 2
net = LSTMNet(input_size, hidden_size, num_layers).to(DEVICE)
trainloader, testloader = load_data(partition_id=partition_id)

# Train
train(net, trainloader, 10)
loss, acc = test(net, testloader)

print('loss: ', loss)
print('acc: ', acc)

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
        train(net, trainloader, epochs=50)
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
