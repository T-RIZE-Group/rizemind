import pandas as pd
import numpy as np


class CityPriceDataset:
    def __init__(self, data_file, feature_scaler, target_scaler, seq_length):
        self.data_path = data_file
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.seq_length = seq_length
        self.data = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df = df.sample(frac=0.3)
        df = df.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code'])
        df = df.dropna()
        X = df.drop(columns='Price').values
        y = df["Price"].values
        X = self.feature_scaler.fit_transform(X)
        y = self.target_scaler.fit_transform(y.reshape(-1, 1))
        
        # Convert to sequences with seq_length
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i+self.seq_length])
            y_seq.append(y[i+self.seq_length])
        
        return np.array(X_seq), np.array(y_seq)

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return len(self.data[0])
