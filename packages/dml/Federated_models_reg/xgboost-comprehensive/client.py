import warnings
from logging import INFO

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
import pandas as pd
from sklearn.model_selection import train_test_split


from dataset import (
    instantiate_partitioner,
    train_test_split_v2,
    transform_dataset_to_dmatrix,
    resplit,
    load_partition, 
    partitioning
)
from utils import client_args_parser, BST_PARAMS, NUM_LOCAL_ROUND
from client_utils import XgbClient


warnings.filterwarnings("ignore", category=UserWarning)


# Parse arguments for experimental settings
args = client_args_parser()

# Train method (bagging or cyclic)
train_method = args.train_method

# Load (HIGGS) dataset and conduct partitioning
# Instantiate partitioner from ["uniform", "linear", "square", "exponential"]
# partitioner = instantiate_partitioner(
#     partitioner_type=args.partitioner_type, num_partitions=args.num_partitions
# )
# fds = FederatedDataset(
#     dataset="jxie/higgs",
#     partitioners={"train": partitioner},
#     resplitter=resplit,
# )

housing_dataset = pd.read_csv(f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data/subset_{args.partition_id}.csv')

print(f'client number {args.partition_id} holds the data of {housing_dataset["City"].iloc[0]}')
print()
city  = housing_dataset['City'].unique()
print(f'city for {args.partition_id} is {city}')
housing_dataset = housing_dataset.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])

housing_dataset = housing_dataset.dropna()
# unique_values = housing_dataset['ocean_proximity'].unique()
# value_to_number = {value: idx for idx, value in enumerate(unique_values)}

# # Replace unique values with numbers
# housing_dataset['ocean_proximity'] = housing_dataset['ocean_proximity'].map(value_to_number)

# # Ensure all columns are numeric and drop rows with missing values
# housing_dataset = housing_dataset.apply(pd.to_numeric, errors='coerce').dropna()
# X = housing_dataset.drop(columns='median_house_value')
# y = housing_dataset['median_house_value']
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.15,
#                                                                                 random_state = 42)


# X_train['median_house_value'] = y_train
partition = housing_dataset
# # all_partitions = partitioning(X_train_global, args.pool_size)
# all_partitions = partitioning(X_train_global, 4)
# Load the partition for this `partition_id`
log(INFO, "Loading partition...")
log(INFO, args.partition_id)
# partition = fds.load_partition(partition_id=args.partition_id, split="train")
# partition.set_format("numpy")

# partition = load_partition(all_partitions, args.partition_id)
log(INFO, 'partition_data: ')
log(INFO, partition.head())

if args.centralised_eval:
    # Use centralised test set for evaluation
    print('client side: centralised eval is activated')
    inputs = partition.drop(columns='Price')
    label = partition['Price']
    train_data = dict()
    train_data['inputs'] = inputs
    train_data['label'] = label
    # train_data = partition
    # valid_data = fds.load_split("test")
    # valid_data.set_format("numpy")
    global_test_data = pd.read_csv('/home/iman/projects/kara/Projects/T-Rize/archive/City_data/global_test_data.csv')
    print('city', city[0])
    global_test_data = global_test_data[global_test_data["City"] == city[0]]
    print(f'central test data of client number {args.partition_id} includes data of {global_test_data["City"].iloc[0]}')
    global_test_data = global_test_data.drop(columns=['Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude'])
    global_test_data = global_test_data.dropna()
    valid_data = dict()
    valid_data['inputs'] = global_test_data.drop(columns='Price')
    valid_data['label'] = global_test_data['Price']
    num_train = train_data['inputs'].shape[0]
    num_val = valid_data['inputs'].shape[0]
else:
    # Train/test splitting
    # train_data, valid_data, num_train, num_val = train_test_split(
    #     partition, test_fraction=args.test_fraction, seed=args.seed
    # )
    inputs = partition.drop(columns='Price')
    label = partition['Price']
    x_train, x_valid, y_train, y_valid = train_test_split(inputs, label, test_size = 0.15,
                                                                                random_state = 42)
    valid_data = dict()
    valid_data['inputs'] = x_valid
    valid_data['label'] = y_valid
    train_data = dict()
    train_data['inputs'] = x_train
    train_data['label'] = y_train
    num_train = train_data['inputs'].shape[0]
    num_val = valid_data['inputs'].shape[0]

# Reformat data to DMatrix for xgboost
log(INFO, "Reformatting data...")
train_dmatrix = transform_dataset_to_dmatrix(train_data)
valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

# Hyper-parameters for xgboost training
num_local_round = NUM_LOCAL_ROUND
params = BST_PARAMS

# Setup learning rate
if args.train_method == "bagging" and args.scaled_lr:
    new_lr = params["eta"] / args.num_partitions
    params.update({"eta": new_lr})

# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=XgbClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
        train_method,
    ),
)
