import warnings
from logging import INFO

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
import pandas as pd

from dataset import (
    instantiate_partitioner,
    transform_dataset_to_dmatrix,
    resplit,
)
from utils import client_args_parser, BST_PARAMS, NUM_LOCAL_ROUND
from client_utils import XgbClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler




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

dataset = pd.read_csv(f'/home/iman/projects/kara/Projects/T-Rize/archive/City_data_classification/subset_{args.partition_id}.csv')
print()
unique_city = dataset['City'].unique()
print(f'Client number {args.partition_id} holds the data of {unique_city}')
print()
dataset = dataset.drop(columns=['Unnamed: 0', 'Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude', 'Price'])
dataset = dataset.dropna()
city_num = dataset.iloc[0]['city_num']
print()
print(f'city code in the partition number {args.partition_id} is: {city_num}')
print()
partition = dataset

# Load the partition for this `partition_id`
log(INFO, "Loading partition...")
# partition = fds.load_partition(partition_id=args.partition_id, split="train")
# partition.set_format("numpy")

# partition_train, partition_test = train_test_split(partition, test_size=0.2)

if args.centralised_eval:
    # Use centralised test set for evaluation
    print('client side: centralised eval is activated')
    inputs = partition.drop(columns='price_label')
    label = partition['price_label']
    train_data = dict()
    train_data['inputs'] = inputs
    train_data['label'] = label
    
    scaler = StandardScaler()
    train_data['inputs'] = scaler.fit_transform(train_data['inputs'])
    
    # train_data = partition
    # valid_data = fds.load_split("test")
    # valid_data.set_format("numpy")
    global_test_data = pd.read_csv('/home/iman/projects/kara/Projects/T-Rize/archive/City_data_classification/global_test_data.csv')
    print('city', unique_city[0])
    global_test_data = global_test_data[global_test_data["City"] == unique_city[0]]
    print('global test data in client side')
    print(global_test_data.head())
    print()
    print(f'central test data of client number {args.partition_id} includes data of {global_test_data["City"].iloc[0]}')
    global_test_data = global_test_data.drop(columns=['Unnamed: 0', 'Unnamed: 0.1','Address', 'City', 'State', 'County', 'Zip Code', 'Latitude', 'Longitude', 'Price'])
    global_test_data = global_test_data.dropna()
    print()
    print('global test data in client side')
    print(global_test_data.head())
    print()
    valid_data = dict()
    valid_data['inputs'] = global_test_data.drop(columns='price_label')
    valid_data['inputs'] = scaler.transform(valid_data['inputs'])
    valid_data['label'] = global_test_data['price_label']
    num_train = train_data['inputs'].shape[0]
    num_val = valid_data['inputs'].shape[0]
else:
    # Train/test splitting
    partition_train, partition_test = train_test_split(partition, test_size=0.2)
    
    inputs = partition_train.drop(columns='price_label')
    label = partition_train['price_label']
    train_data = dict()
    train_data['inputs'] = inputs
    train_data['label'] = label
    scaler = StandardScaler()
    train_data['inputs'] = scaler.fit_transform(train_data['inputs'])
    
    input_valid = partition_test.drop(columns='price_label')
    label_valid = partition_test['price_label']
    valid_data = dict()
    valid_data['inputs'] = input_valid
    valid_data['label'] = label_valid
    valid_data['inputs'] = scaler.transform(valid_data['inputs'])
    
    num_train = partition_train.shape[0]
    num_val = partition_test.shape[0]
    
    # train_data, valid_data, num_train, num_val = train_test_split(
    #     partition, test_fraction=args.test_fraction, seed=args.seed
    # )

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
