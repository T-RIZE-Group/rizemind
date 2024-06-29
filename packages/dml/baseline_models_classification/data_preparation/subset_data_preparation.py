import pandas as pd
from sklearn.model_selection import train_test_split
import json

def data_splitting_2(file_path, data_root_path, testsize):
    # Load the dataset
    # file_path = 'path_to_your_csv_file.csv'  # Replace with your file path
    dataset = pd.read_csv(file_path)
    
    
    # Create a mapping from county names to unique numbers
    
    unique_counties = dataset['County'].unique()

    
    county_to_number = {county: idx for idx, county in enumerate(unique_counties)}
    
    with open(f'{data_root_path}/county_mapping.json', 'w') as json_file:
        json.dump(county_to_number, json_file)

    
    dataset['county_num'] = dataset['County'].map(county_to_number)
    
    # Create a mapping from city names to unique numbers
    
    unique_cities = dataset['City'].unique()

    
    City_to_number = {city: idx for idx, city in enumerate(unique_cities)}
    
    with open(f'{data_root_path}/city_mapping.json', 'w') as json_file:
        json.dump(City_to_number, json_file)

    
    dataset['city_num'] = dataset['City'].map(City_to_number)
    
    # Create a mapping from state names to unique numbers
    
    
    unique_states = dataset['State'].unique()

    
    State_to_number = {state: idx for idx, state in enumerate(unique_states)}
    
    with open(f'{data_root_path}/state_mapping.json', 'w') as json_file:
        json.dump(State_to_number, json_file)

    
    dataset['state_num'] = dataset['State'].map(State_to_number)
    
    # Create a mapping from address names to unique numbers
    
    unique_address = dataset['Address'].unique()

    
    address_to_number = {address: idx for idx, address in enumerate(unique_address)}
    
    with open(f'{data_root_path}/address_mapping.json', 'w') as json_file:
        json.dump(address_to_number, json_file)

    
    dataset['address_num'] = dataset['Address'].map(address_to_number)
    
    global_trainData, global_testData = train_test_split(dataset, test_size=testsize)
    global_testData.to_csv(f'{data_root_path}/global_test_data.csv', index=False)

    # Extract unique city names
    unique_cities = global_trainData['City'].unique()

    # Create a dictionary to hold dataframes for each city
    city_datasets = {city: global_trainData[global_trainData['City'] == city] for city in unique_cities}

    # Save each city's dataset into separate CSV files
    index = 0
    for city, df in city_datasets.items():
        city_name = city.replace(' ', '_')  # Replace spaces with underscores for file names
        if len(df) >= 100:
            df.to_csv(f'{data_root_path}/subset_{index}.csv', index=False)
            index = index + 1

    # Print the unique cities and the corresponding number of rows for each subdataset
    for city, df in city_datasets.items():
        print(f'City: {city}, Number of rows: {len(df)}')

dataset = '/home/iman/projects/kara/Projects/T-Rize/archive/American_housing_data.csv'
data_root_path = '/home/iman/projects/kara/Projects/T-Rize/archive/City_data_classification'
data_splitting_2( dataset, data_root_path, 0.2)