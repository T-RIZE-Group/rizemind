import flwr as fl
from functools import partial
from cnn_model import CNN  # Make sure CNN is defined in cnn_model.py
from client import create_client, weighted_average, NUM_CLIENTS  # Import from client.py

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
    min_fit_clients=2,  # Never sample less than 2 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=5,  # Wait until all 5 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,
)

client_fnc = partial(
    create_client,
    model_class=CNN,
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fnc,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0},
    ray_init_args={
        "num_cpus": 1,
        "num_gpus": 0,
        "_system_config": {"automatic_object_spilling_enabled": False},
    },
)
