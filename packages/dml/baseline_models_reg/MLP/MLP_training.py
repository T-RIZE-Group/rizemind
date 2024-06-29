import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 16)
        self.fc9 = nn.Linear(16, 8)
        self.fc10 = nn.Linear(8, 8)
        self.fc11 = nn.Linear(8, 4)
        self.fc12 = nn.Linear(4, 4)
        self.fc13 = nn.Linear(4, 2)
        self.fc14 = nn.Linear(2, 2)
        self.fc15 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        x = torch.relu(self.fc9(x))
        x = torch.relu(self.fc10(x))
        x = torch.relu(self.fc11(x))
        x = torch.relu(self.fc12(x))
        x = torch.relu(self.fc13(x))
        x = torch.relu(self.fc14(x))
        x = self.fc15(x)
        return x

# Load your dataset
num_subsets = 4
dataframes = []
for i in range(num_subsets):
    file_paths = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{i}.csv'
    df = pd.read_csv(file_paths)
    dataframes.append(df)

# Concatenate all dataframes to create one dataset
dataset = pd.concat(dataframes, ignore_index=True)

unique_cities = dataset['City'].unique()

dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])

dataset = dataset.dropna()
X = dataset.drop(columns='Price').values
y = dataset["Price"].values

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add a dimension for single output

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Add a dimension for single output

# Initialize model
model = MLPRegressor(input_dim=X_train.shape[1])

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 10
batch_size = 32

for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]

        # Forward pass
        outputs = model(batch_X)

        # Compute loss
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    # Make predictions on the test set
    predicted = model(X_test_tensor)

    # Calculate MSE
    mse = mean_squared_error(y_test_tensor.numpy(), predicted.numpy())
    print(f'Mean Squared Error (MSE): {mse:.4f}')

    # Calculate R2 score
    r2 = r2_score(y_test_tensor.numpy(), predicted.numpy())
    print(f'R2 Score: {r2:.4f}')
