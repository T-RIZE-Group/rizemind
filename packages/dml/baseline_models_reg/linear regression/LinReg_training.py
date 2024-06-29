import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import pickle 

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
inputs = dataset.drop(columns='Price')
label = dataset["Price"]

x_train, x_test, y_train, y_test = train_test_split(inputs, label, test_size = 0.3,
                                                                                random_state = 42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred_test = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print('unique cities: ', unique_cities)
print('dataset length = ', len(x_train))
print('num features: ', x_train.shape[1])
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

with open('/home/iman/projects/kara/Projects/T-Rize/baseline_models_training/linear regression/linear_reg_model.pkl', 'wb') as file:
    pickle.dump(model, file)