import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Load and preprocess your data
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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (optional, but can improve performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the data to DMatrix, which is a data structure used by XGBoost
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Set the parameters for the XGBoost classifier
params = {
    'objective': 'multi:softmax',  # Specify the objective for multi-class classification
    'num_class': len(np.unique(y)),  # Number of classes
    'eval_metric': 'mlogloss'  # Evaluation metric for multi-class classification
}

# Train the XGBoost model
num_rounds = 100  # Number of boosting rounds
bst = xgb.train(params, dtrain, num_rounds)

# Make predictions on the test set
y_test_pred = bst.predict(dtest)

# Evaluate the classifier
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(classification_report(y_test, y_test_pred))

# Save the model
bst.save_model('xgboost_model.json')
print("Model saved to 'xgboost_model.json'")
