import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler



dataset = pd.read_csv('/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_5.csv')
dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code'])



inputs = dataset.drop(columns='Price')
label = dataset["Price"]

x_train, x_test, y_train, y_test = train_test_split(inputs, label, test_size = 0.15,
                                                                                random_state = 42)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

# Make predictions
y_pred = bst.predict(dtest)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('dataset length = ', len(dataset))
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

print()