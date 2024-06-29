import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM
from keras.optimizers import Adam

num_subsets = 4
dataframes = []
for i in range(num_subsets):
    file_path = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data_classification/subset_{i}.csv'
    df = pd.read_csv(file_path)
    dataframes.append(df)

# Step 3: Concatenate all dataframes in the list to create one dataset
dataset = pd.concat(dataframes, ignore_index=True)

unique_cities = dataset['City'].unique()

# Add price_label column
# price_bins = [0, 100000, 200000, 300000, 400000, 500000, np.inf]
# price_labels = [0, 1, 2, 3, 4, 5]
# dataset['price_label'] = pd.cut(dataset['Price'], bins=price_bins, labels=price_labels)

num_classes = len(dataset['price_label'].unique())
print('num of classes: ', num_classes)

# Dropping unwanted columns
dataset = dataset.drop(columns=['Unnamed: 0', 'Address', 'City', 'State', 'County', 'Zip Code', 'Price'])

dataset = dataset.dropna()

X = dataset.drop(columns='price_label').values
y = dataset['price_label'].values

# Convert to numpy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model with improved architecture
model = Sequential([
    Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(128, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Conv1D(256, kernel_size=3, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape input data for Conv1D
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Training
num_epochs = 10
batch_size = 32

history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Predictions and evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Score:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Print final accuracy
print(f'Final Overall Accuracy: {accuracy:.4f}')

print('dataset length: ', len(X_train))
print('num features: ', X_train.shape[1])
print('unique cities: ', unique_cities)
