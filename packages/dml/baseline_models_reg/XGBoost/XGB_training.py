import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


num_subsets = 4
dataframes = []
for i in range(num_subsets):
    file_paths = f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{i}.csv'
    df = pd.read_csv(file_paths)
    dataframes.append(df)

# Step 3: Concatenate all dataframes in the list to create one dataset
dataset = pd.concat(dataframes, ignore_index=True)

# file_path = '/home/iman/projects/kara/Projects/T-Rize/archive/American_Housing_Data_20231209.csv'


# dataset = pd.read_csv(file_path)
    
    
    # Create a mapping from county names to unique numbers

dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])

dataset = dataset.dropna()
inputs = dataset.drop(columns='Price')
label = dataset["Price"]

x_train, x_test, y_train, y_test = train_test_split(inputs, label, test_size = 0.3,
                                                                                random_state = 42)
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

# Set parameters for XGBoost
params = {
    'objective': 'reg:squarederror',  # Regression with squared error
    'max_depth': 6,                   # Maximum depth of a tree
    'eta': 0.3,                       # Learning rate
    'eval_metric': 'rmse'             # Evaluation metric
}

# Train the model
num_round = 100  # Number of boosting rounds
bst = xgb.train(params, dtrain, num_round)

bst.save_model('/home/iman/projects/kara/Projects/T-Rize/baseline_models_training/XGBoost/XGBoost_model.json')
# Make predictions
y_pred = bst.predict(dtest)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('dataset length = ', len(x_train))
print('num features: ', x_train.shape[1])
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

print()