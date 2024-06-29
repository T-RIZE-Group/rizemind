import argparse
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf

import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from flwr_datasets import FederatedDataset
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from LSTM_dataPreparation import CityPriceDataset
import optuna




# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
    
    def get_weights(self):
        return self.trainable_weights
    
    def set_weights(self, parameters):
        self.lstm.set_weights(parameters)
    




def load_partition(idx: int):
    """"""
    # Download and partition dataset
    data_dir = "/home/iman/projects/kara/Projects/T-Rize/archive/City_data"
    data_file = f'subset_{idx}.csv' 

    X = []
    y = []

    # Define scalers
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit the scalers on the entire dataset
   
    full_path = os.path.join(data_dir, data_file)
    df = pd.read_csv(full_path)
    df = df.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code'])
    df = df.dropna()
    feature_scaler.partial_fit(df.drop(columns='Price').values)
    target_scaler.partial_fit(df["Price"].values.reshape(-1, 1))

    # Load the data using the fitted scalers
    
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

    
    
   
    
    return X_train, y_train, X_test, y_test






# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        dummy_input = np.zeros((1, 12, x_train.shape[2]))  # Adjust seq_length as needed
        self.model(dummy_input)
        
    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        print()
        
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        
        y_pred = self.model.predict(self.x_train)
        r2 = r2_score(y_pred, self.y_train)
        
        results = {
            "loss": history.history["loss"][0],
            # "accuracy": history.history["accuracy"][0],
            # "val_loss": r2,
            "val_accuracy": r2,
        }
        print()
        print('accuracy on the client side after fit: ')
        print('accuracy: ', results['val_accuracy'])
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        # loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        y_pred = self.model.predict(self.x_test)
        loss = mean_squared_error(y_pred, self.y_test)
        accuracy = r2_score(y_pred, self.y_test)
        print()
        print('accuracy on the client side in evaluate function')
        print('accuracy: ')
        print(accuracy)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--client-id",
        type=int,
        default=0,
        choices=range(0, 10),
        required=True,
        help="Specifies the artificial data partition of CIFAR10 to be used. "
        "Picks partition 0 by default",
    )
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Set to true to quicky run the client using only 10 datasamples. "
        "Useful for testing purposes. Default: False",
    )
    args = parser.parse_args()

    # Load and compile Keras model
    input_dim = 10
    # num_classes = 11
    x_train, y_train, x_test, y_test = load_partition(args.client_id)
    
    
    input_size = x_train[2]
    
    model = LSTMNet(input_size, 72, 3,
                0.273, seq_length=12)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.006),
              loss='mse')
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=30, batch_size=64,  # Set epochs to 30 for the final training
                        validation_split=0.2, verbose=1, callbacks=[early_stopping])


  

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test).to_client()

    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
        root_certificates=Path("/home/iman/projects/kara/Projects/T-Rize/federated_models_classification/MLP/.cache/certificates/ca.crt").read_bytes(),
    )




if __name__ == "__main__":
    main()
