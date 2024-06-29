import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Read and preprocess data
num_subsets = 4
dataframes = []
for i in range(num_subsets):
    file_path = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{i}.csv'
    df = pd.read_csv(file_path)
    dataframes.append(df)

dataset = pd.concat(dataframes, ignore_index=True)

unique_cities = dataset['City'].unique()
dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code'])
dataset = dataset.dropna()

X = dataset.drop(columns='Price').values
y = dataset["Price"].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Reshape data to match Conv1D input requirements
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Build the model
model = Sequential()
model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2)) 
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.2))  
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(256, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.2)) 
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

model.add(LSTM(128, return_sequences=True))  
model.add(Flatten())
model.add(Dropout(0.5))  
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
num_epochs = 20  
batch_size = 32

history = model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Loss: {test_loss:.4f}')

# Make predictions
y_pred = model.predict(X_test)

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.4f}')

# Calculate R2 score
r2 = r2_score(y_test, y_pred)
print(f'R2 Score: {r2:.4f}')

print('dataset length: ', len(X_train))    
print('num features: ', X_train.shape[1])
print('unique cities: ', unique_cities)
