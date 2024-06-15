
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.model_selection import train_test_split


dataset = pd.read_csv(f'/home/iman/projects/kara/Projects/T-Rize/sklearn-logreg-mnist/City_data/subset_{0}.csv')
print()
dataset = dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code'])



X = dataset.drop(columns='Price')
y = dataset['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15,
                                                                            random_state = 42)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

model = Ridge()

model.fit(X_train, y_train)

print(model.coef_)

print()