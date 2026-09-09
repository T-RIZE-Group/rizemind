import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(num_clients):
    # Generate synthetic binary classification dataset
    X, y = make_classification(n_samples=num_clients * 1000 + 2000, n_features=20, n_classes=2, random_state=42)
    
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=2000, random_state=42)
    
    # 1000 for validation (task utility), 1000 for probe set (attacker features)
    X_val, X_probe, y_val, y_probe = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Split training data evenly among clients
    client_data = []
    chunk_size = len(X_train) // num_clients
    for i in range(num_clients):
        X_c = X_train[i*chunk_size:(i+1)*chunk_size]
        y_c = y_train[i*chunk_size:(i+1)*chunk_size]
        dataset = TensorDataset(torch.tensor(X_c), torch.tensor(y_c).unsqueeze(1))
        client_data.append(DataLoader(dataset, batch_size=32, shuffle=True))
        
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    probe_dataset = TensorDataset(torch.tensor(X_probe), torch.tensor(y_probe).unsqueeze(1))
    probe_loader = DataLoader(probe_dataset, batch_size=32, shuffle=False)
    
    return client_data, val_loader, probe_loader
