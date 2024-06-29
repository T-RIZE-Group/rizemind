import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
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


num_subsets = 4
dataframes = []
for i in range(num_subsets):
    file_paths = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{i}.csv'
    df = pd.read_csv(file_paths)
    dataframes.append(df)

# Step 3: Concatenate all dataframes in the list to create one dataset
dataset = pd.concat(dataframes, ignore_index=True)

unique_cities = dataset['City'].unique()

dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])

dataset = dataset.dropna()
X = dataset.drop(columns='Price').values
y = dataset["Price"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Adding a dimension for single output

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Net(num_features=X.shape[1])

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 10
batch_size = 32


for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(batch_X.unsqueeze(1))  # Add channel dimension

        # Compute loss
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()

with torch.no_grad():
    predicted = model(X_test.unsqueeze(1))  # Add channel dimension
    test_loss = criterion(predicted, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

    # Calculate MSE
    mse = mean_squared_error(y_test.numpy(), predicted.numpy())
    print(f'Mean Squared Error (MSE): {mse:.4f}')

    # Calculate R2 score
    r2 = r2_score(y_test.numpy(), predicted.numpy())
    print(f'R2 Score: {r2:.4f}')
    
print('dataset length: ', len(X_train))    
print('num features: ', X_train.shape[1])
print('unique cities: ', unique_cities)