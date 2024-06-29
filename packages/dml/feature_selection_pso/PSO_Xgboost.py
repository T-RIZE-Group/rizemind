import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
import numpy as np

def train_xgboost(dataset_file):
    print(f"Training XGBoost model on dataset: {dataset_file}")

    # Read the dataset
    dataset = pd.read_csv(dataset_file)

    # Remove unwanted columns
    dataset = dataset.drop(columns=['Unnamed: 0', 'Latitude', 'Longitude', 'Price'])

    # Identify target and numeric columns
    target_column = 'price_label'
    numeric_columns = dataset.select_dtypes(include=np.number).columns.tolist()

    # Drop non-numeric columns if exist
    dataset_numeric = dataset[numeric_columns]

    # Prepare data for modeling
    X = dataset_numeric.drop(columns=[target_column]).values
    y = dataset_numeric[target_column].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create XGBoost model
    model = XGBRegressor()

    # Use SelectFromModel to select features based on importance
    selector = SelectFromModel(estimator=model, threshold=-np.inf, max_features=4)
    selector.fit(X_train, y_train)

    # Transform training and test datasets
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Train model with selected features
    model.fit(X_train_selected, y_train)

    # Calculate overall accuracy of XGBoost model on test set
    overall_accuracy = model.score(X_test_selected, y_test)
    print(f"Overall accuracy of XGBoost model on test set: {overall_accuracy}")

    # Return selected feature names, selector, and the model
    selected_features_names = dataset_numeric.drop(columns=[target_column]).columns[selector.get_support(indices=True)]
    return selected_features_names, selector, model
