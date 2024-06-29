import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pyswarm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the parametrized CNN model
class ParamNet(nn.Module):
    def __init__(self, input_dim, conv_params, fc_params, dropout_rate=0.5):
        super(ParamNet, self).__init__()
       
        self.conv1 = nn.Conv1d(1, conv_params[0], kernel_size=conv_params[1], padding=1)
        self.bn1 = nn.BatchNorm1d(conv_params[0])
        self.conv2 = nn.Conv1d(conv_params[0], conv_params[2], kernel_size=conv_params[3], padding=1)
        self.bn2 = nn.BatchNorm1d(conv_params[2])
        self.conv3 = nn.Conv1d(conv_params[2], conv_params[4], kernel_size=conv_params[5], padding=1)
        self.bn3 = nn.BatchNorm1d(conv_params[4])
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout_rate)
       
        # Calculate the output dimension after conv and pooling layers
        conv_output_dim = self.calculate_conv_output_dim(input_dim, conv_params)
        fc_input_dim = conv_output_dim * conv_params[4]
       
        self.fc1 = nn.Linear(fc_input_dim, fc_params[0])
        self.fc2 = nn.Linear(fc_params[0], fc_params[1])
        self.fc3 = nn.Linear(fc_params[1], 1)
       
    def calculate_conv_output_dim(self, input_dim, conv_params):
        output_dim = input_dim
        for i in range(3):  # 3 conv layers
            output_dim = (output_dim + 2 - (conv_params[2*i+1] - 1) - 1) // 2 + 1
        return output_dim

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def create_model(input_dim, conv_params, fc_params, dropout_rate=0.5):
    return ParamNet(input_dim, conv_params, fc_params, dropout_rate)

def train_and_evaluate(conv_params, fc_params, dropout_rate, X_train, y_train, X_test, y_test, num_epochs=10, batch_size=32, learning_rate=0.001):
    input_dim = X_train.shape[1]
    model = create_model(input_dim, conv_params, fc_params, dropout_rate)
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
   
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)
   
    for epoch in range(num_epochs):
        model.train()
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            outputs = model(batch_X.unsqueeze(1))
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
    model.eval()
    with torch.no_grad():
        predicted = model(X_test_tensor.unsqueeze(1))
        mse = mean_squared_error(y_test_tensor.cpu().numpy(), predicted.cpu().numpy())
       
    return mse

# Load and preprocess your dataset
num_subsets = 4
dataframes = []
for i in range(num_subsets):
    file_paths = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{i}.csv'
    df = pd.read_csv(file_paths)
    dataframes.append(df)

dataset = pd.concat(dataframes, ignore_index=True)
dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])
dataset = dataset.dropna()

X = dataset.drop(columns='Price').values
y = dataset["Price"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# PSO parameters
max_kernel_size = min(X_train.shape[1] // 4, 7)  # Ensure kernel size is reasonable
lb = [16, 3, 32, 3, 64, 3, 128, 256]  # Set lower bounds for [conv1_filters, conv1_kernel, conv2_filters, conv2_kernel, conv3_filters, conv3_kernel, fc1_units, fc2_units]
ub = [64, max_kernel_size, 128, max_kernel_size, 256, max_kernel_size, 512, 512]  # Set upper bounds for the same parameters

# Ensure upper bounds are greater than lower bounds
ub = [max(lb[i] + 1, ub[i]) for i in range(len(lb))]

def pso_objective(params):
    conv_params = [
        int(params[0]), min(int(params[1]), X_train.shape[1]),  
        int(params[2]), min(int(params[3]), X_train.shape[1] // 2),  
        int(params[4]), min(int(params[5]), X_train.shape[1] // 4)  
    ]
    fc_params = [
        int(params[6]),  
        int(params[7])  
    ]
    dropout_rate = 0.5
    mse = train_and_evaluate(conv_params, fc_params, dropout_rate, X_train, y_train, X_test, y_test)
    return mse

# Run PSO
best_params, best_mse = pyswarm.pso(pso_objective, lb, ub, swarmsize=10, maxiter=10)
print(f'Best Parameters: {best_params}')
print(f'Best MSE: {best_mse}')
