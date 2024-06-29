import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from PSO_XGBoost_Regressor import PSO_XGBoost_regressor

# List of CSV files
file_paths = [
    '/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_0.csv',
    '/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_1.csv',
    '/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_2.csv',
    '/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_3.csv'
]

# Loading data and combining into one dataframe
dfs = [pd.read_csv(file) for file in file_paths]
dataset = pd.concat(dfs, ignore_index=True)

# Drop 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in dataset.columns:
    dataset = dataset.drop(columns=['Unnamed: 0'])

# Drop 'price' column if it exists
column_to_drop = 'price'
if column_to_drop in dataset.columns:
    dataset = dataset.drop(columns=[column_to_drop])

# Extracting features and target
dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])
X = dataset.drop(columns=['Price'])  # Features
y = dataset['Price']  # Target

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creating and using PSO_XGBoost_regressor model
popSize = 10
maxIt = 75
regressor = PSO_XGBoost_regressor(x_train, y_train, popSize, maxIt)
regressor.fit(isPlot=True)

# Predicting on test data
y_pred = regressor.predict(x_test)

# Calculating evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print('best cost: ', regressor.gbestCost)
print('best solution: ', regressor.gbestPos)
print('r2 score: ', r2, ' mean squared error: ', mse)
