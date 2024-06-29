import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow as keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import os
import optuna

# Define LSTM network
class LSTMNet(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_layers, dropout, seq_length):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = LSTM(hidden_size, return_sequences=False, input_shape=(seq_length, input_size), dropout=dropout)
        self.fc1 = Dense(128)
        self.bn1 = BatchNormalization()
        self.fc2 = Dense(1)

    def call(self, x):
        x = self.lstm(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x

# Define custom Dataset class
class CityPriceDataset:
    def __init__(self, data_file, feature_scaler, target_scaler, seq_length):
        self.data_path = data_file
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.seq_length = seq_length
        self.data = self.load_data()

    def load_data(self):
        df = pd.read_csv(self.data_path)
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

# Load and preprocess data
data_dir = "/home/iman/projects/kara/Projects/T-Rize/archive/City_data"
data_files = [f'subset_{i}.csv' for i in range(4)]

X = []
y = []

# Define scalers
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit the scalers on the entire dataset
for data_file in data_files:
    full_path = os.path.join(data_dir, data_file)
    df = pd.read_csv(full_path)
    df = df.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code'])
    df = df.dropna()
    feature_scaler.partial_fit(df.drop(columns='Price').values)
    target_scaler.partial_fit(df["Price"].values.reshape(-1, 1))

# Load the data using the fitted scalers
for data_file in data_files:
    full_path = os.path.join(data_dir, data_file)
    dataset = CityPriceDataset(full_path, feature_scaler, target_scaler, seq_length=12)  # Adjust seq_length as needed
    X_data, y_data = dataset[:]
    X.append(X_data)
    y.append(y_data)

X = np.concatenate(X)
y = np.concatenate(y)

# Get input size for the LSTM model
input_size = X.shape[2]  # Adjust based on the shape of X

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Define hyperparameters optimization objective function
def objective(trial):
    # Hyperparameters to tune
    hidden_size = trial.suggest_int("hidden_size", 64, 512)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)

    # Create model
    model = LSTMNet(input_size, hidden_size, num_layers, dropout, seq_length=12)  # Adjust seq_length as needed

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse')

    # Split data into training and validation sets
    X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Early Stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train_fold, y_train_fold, epochs=10, batch_size=64,  # Adjust epochs to balance the total number
                        validation_data=(X_val_fold, y_val_fold), verbose=1, callbacks=[early_stopping])

    avg_val_loss = np.mean(history.history['val_loss'])
    print(f"Avg Validation Loss: {avg_val_loss}")

    return avg_val_loss

# Run hyperparameter optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)  # Reduced number of trials to 5

# Print the best hyperparameters
best_params = study.best_trial.params
print(f"Best trial: {study.best_trial.value}")
print(f"Best hyperparameters: {best_params}")

# Train the final model with the best hyperparameters
model = LSTMNet(input_size, best_params["hidden_size"], best_params["num_layers"],
                best_params["dropout"], seq_length=12)  # Adjust seq_length as needed

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"]),
              loss='mse')

# Train the final model
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=30, batch_size=64,  # Set epochs to 30 for the final training
                    validation_split=0.2, verbose=1, callbacks=[early_stopping])

# Evaluate the final model on the test set
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss}")

# Calculate R2 score
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R2 Score: {r2}")
