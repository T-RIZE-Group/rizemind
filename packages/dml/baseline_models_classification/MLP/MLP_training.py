
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

def create_mlp_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(200, input_dim=input_dim, activation='relu'))  # First hidden layer
    model.add(Dense(200, activation='relu'))  # Second hidden layer
    model.add(Dense(200, activation='relu'))  # Third hidden layer
    model.add(Dense(num_classes, activation='softmax'))
    return model


num_subsets = 4
dataFrames = []
for i in range(num_subsets):
    data_path = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data_classification/subset_{i}.csv'
    df = pd.read_csv(data_path)
    dataFrames.append(df)

dataset = pd.concat(dataFrames, ignore_index=True)

unique_cities = dataset['City'].unique()

dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude', 'Price'])

dataset = dataset.dropna()
X = dataset.drop(columns='price_label').values
y = dataset["price_label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_dim = X_train.shape[1]
num_classes = len(dataset["price_label"].unique())
print('num classes: ', num_classes)
hidden_layers = 3  # Adjust the number of hidden layers here
neurons_per_layer = 64  # Adjust the number of neurons per layer here

y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

model = create_mlp_model(input_dim, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_categorical, epochs=50, batch_size=32, validation_split=0.2)


loss, accuracy = model.evaluate(X_test, y_test_categorical)
print(f'Test Accuracy: {accuracy:.4f}')
