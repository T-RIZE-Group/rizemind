import torch
from CNN_class import Net
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

num_subsets = 4
dataframes = []
for i in range(num_subsets):
    file_paths = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{i}.csv'
    df = pd.read_csv(file_paths)
    dataframes.append(df)

# Step 3: Concatenate all dataframes in the list to create one dataset
dataset = pd.concat(dataframes, ignore_index=True)

unique_cities = dataset['City'].unique()

dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])

dataset = dataset.dropna()
X = dataset.drop(columns='Price')
y = dataset["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

########## for CNN

X_test_cnn = torch.tensor(X_test.values, dtype=torch.float32)
# y_test_cnn = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1) 
CNN_model = Net(X_train.shape[1])

with torch.no_grad():
    predicted_CNN = CNN_model(X_test_cnn.unsqueeze(1))  # Add channel dimension


predicted_CNN = predicted_CNN.numpy()
########## for XGBoost

XGBoost_model = xgb.Booster()
XGBoost_model.load_model('/home/iman/projects/kara/Projects/T-Rize/baseline_models_training/Ensemble_baselines/XGBoost_model.json')

dtest = xgb.DMatrix(X_test)

predicted_XGBoost = XGBoost_model.predict(dtest)




########## for linear regression

with open('/home/iman/projects/kara/Projects/T-Rize/baseline_models_training/Ensemble_baselines/linear_reg_model.pkl', 'rb') as file:
    linear_reg_model = pickle.load(file)


predicted_LR = linear_reg_model.predict(X_test)

print()

############# ensembling

Ensemble_result = []

for i in range(len(y_test)):
    Ensemble_result.append((predicted_CNN[i] + predicted_LR[i] + predicted_XGBoost[i])/3)


r2 = r2_score(Ensemble_result, y_test)
mse = mean_squared_error(Ensemble_result, y_test)

print('r2: ', r2, 'mse: ', mse)

print('predicted_cnn[0]', predicted_CNN[0])
print('predicted_XGBoost[0]', predicted_XGBoost[0])
print('predicted_lr[0]', predicted_LR[0])
print('ensemble_result[0]', Ensemble_result[0])
       