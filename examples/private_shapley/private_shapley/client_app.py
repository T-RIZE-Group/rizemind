"""Client application for the Private Shapley example."""

import os
import hashlib
from typing import Dict, List, Tuple, cast, Optional
import torch
from dotenv import load_dotenv
from eth_account import Account
from eth_typing import HexAddress
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, Parameters, Scalar

from .task import Net, get_weights, load_data, set_weights, train, test
from .contract import PrivateShapleyContract
from .commit import CoalitionManager, generate_nonce, create_commitment

# Load environment variables
load_dotenv()

# Global coalition manager
coalition_manager = CoalitionManager()

class PrivateShapleyClient(NumPyClient):
    """Client for the Private Shapley federated learning example."""
    
    def __init__(
        self,
        trainloader,
        testloader,
        epochs: int,
        learning_rate: float,
        trainer_address: str,
        private_key: str,
        contract: PrivateShapleyContract,
    ):
        """Initialize the client.
        
        Args:
            trainloader: DataLoader for training data
            testloader: DataLoader for test data
            epochs: Number of local training epochs
            learning_rate: Learning rate for optimization
            trainer_address: Ethereum address of the trainer
            private_key: Private key for the trainer
            contract: Interface to the PrivateShapley contract
        """
        self.net = Net()
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainer_address = trainer_address
        self.private_key = private_key
        self.contract = contract
        self.trainer_index = None
        self.round_id = 0
        self.round_nonce = None
        
        # Register with the contract if not already registered
        if not contract.is_registered_trainer(trainer_address):
            print(f"Client {trainer_address} not registered. Will be registered by server.")
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List:
        """Return the current local model parameters."""
        return get_weights(self.net)
    
    def fit(self, parameters: List, config: Dict[str, Scalar]) -> Tuple[List, int, Dict[str, Scalar]]:
        """Train the model on the local dataset."""
        # Update round ID
        self.round_id = int(config.get("round_id", 0))
        print(f"Starting training round {self.round_id}")
        
        # Use the trainer address from config if provided
        trainer_address = config.get("trainer_address")
        if trainer_address:
            self.trainer_address = trainer_address
        
        # Check if we have a trainer index
        if self.trainer_index is None and "trainer_index" in config:
            self.trainer_index = int(config["trainer_index"])
            # Register with the coalition manager
            coalition_manager.register_trainer(self.trainer_address, self.trainer_index)
            print(f"Registered as trainer with index {self.trainer_index}")
        
        # Generate a new nonce for this round
        self.round_nonce = coalition_manager.generate_trainer_nonce(
            self.trainer_address, self.round_id
        )
        
        # Set model parameters
        set_weights(self.net, parameters)
        
        # Train the model
        training_metrics = train(
            self.net, self.trainloader, self.epochs, self.learning_rate, self.device
        )
        
        # Get updated parameters
        updated_parameters = get_weights(self.net)
        
        # Create a commitment for this round
        commitment = create_commitment(
            self.trainer_address, self.round_nonce, self.round_id
        )
        
        # Return the updated parameters and additional metrics
        metrics_dict = {
            "trainer_address": self.trainer_address,
            "commitment": commitment.hex(),
        }

        # Add trainer_index only if it's not None
        if self.trainer_index is not None:
            metrics_dict["trainer_index"] = self.trainer_index

        # Add training metrics, ensuring no None values
        for key, value in training_metrics.items():
            if value is not None:  # Skip None values
                metrics_dict[key] = value

        return updated_parameters, len(self.trainloader.dataset), metrics_dict
        
        # # Return the updated parameters and additional metrics
        # return updated_parameters, len(self.trainloader.dataset), {
        #     "trainer_address": self.trainer_address,
        #     "commitment": commitment.hex(),
        #     "trainer_index": self.trainer_index,
        #     **training_metrics
        # }
    
    def evaluate(self, parameters: List, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on the local test set."""
        # Set model parameters
        set_weights(self.net, parameters)
        
        # Test the model
        loss, accuracy = test(self.net, self.testloader, self.device)
        
        # Get coalition ID if provided
        coalition_id = config.get("coalition_id")
        
        # Check if we need to claim rewards
        if coalition_id:
            coalition_id_bytes = bytes.fromhex(cast(str, coalition_id))
            # We'll check if this client is in the coalition and eligible for rewards
            self._check_and_claim_rewards(coalition_id_bytes)
        
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}
    
    # def _check_and_claim_rewards(self, coalition_id: bytes) -> None:
    #     """Check if eligible for rewards and claim if possible."""
    #     try:
    #         if self.trainer_index is None or self.round_nonce is None:
    #             print("Cannot claim rewards - missing trainer index or nonce")
    #             return
            
    #         # Check if this trainer is in the coalition
    #         if not self.contract.is_trainer_in_coalition(coalition_id, self.trainer_address):
    #             print(f"Not in coalition {coalition_id.hex()}, skipping reward claim")
    #             return
            
    #         # Check for new results
    #         result_events = self.contract.get_coalition_result_events()
    #         for event in result_events:
    #             event_coalition_id = event['coalition_id']
    #             if event_coalition_id == coalition_id:
    #                 print(f"Found result for coalition {coalition_id.hex()}")
                    
    #                 # Get the merkle proof
    #                 # This is a simplified example - in practice, we would need to know all coalition members
    #                 # Here we're assuming the server will provide this information
    #                 # For now, we'll skip the actual claiming since we don't have all the information
                    
    #                 print(f"Would claim reward for coalition {coalition_id.hex()} with result {event['result']}")
                    
    #                 # To actually claim the reward:
    #                 # merkle_proof = coalition_manager.generate_merkle_proof(
    #                 #     self.round_id, coalition_members, self.trainer_address
    #                 # )
    #                 # self.contract.claim_reward(
    #                 #     self.round_id, coalition_id, self.round_nonce,
    #                 #     merkle_proof, self.trainer_address, self.private_key
    #                 # )
                    
    #                 return
            
    #         print(f"No results found for coalition {coalition_id.hex()}")
        
    #     except Exception as e:
    #         print(f"Error checking/claiming rewards: {e}")
    
    def _check_and_claim_rewards(self, coalition_id: bytes) -> None:
        """Check if eligible for rewards and claim if possible."""
        try:
            if self.trainer_index is None or self.round_nonce is None:
                print("Cannot claim rewards - missing trainer index or nonce")
                return
            
            # Check if this trainer is in the coalition
            if not self.contract.is_trainer_in_coalition(coalition_id, self.trainer_address):
                print(f"Not in coalition {coalition_id.hex()}, skipping reward claim")
                return
            
            # Check for new results
            result_events = self.contract.get_coalition_result_events()
            for event in result_events:
                event_coalition_id = event['coalition_id']
                if event_coalition_id == coalition_id:
                    print(f"Found result for coalition {coalition_id.hex()} with result {event['result']}")
                    
                    # For now, log that we would claim a reward
                    print(f"Would claim reward for coalition {coalition_id.hex()}")
                    return
            
            print(f"No results found for coalition {coalition_id.hex()}")
        
        except Exception as e:
            print(f"Error checking/claiming rewards: {e}")

def client_fn(context: Context):
    """Create and configure a client instance.
    
    Args:
        context: The client context provided by Flower
        
    Returns:
        A configured client instance
    """
    # Load environment variables
    load_dotenv()
    
    # Get client-specific configuration
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    
    # Get run-specific configuration
    batch_size = int(context.run_config["batch-size"])
    epochs = int(context.run_config["local-epochs"])
    learning_rate = float(context.run_config["learning-rate"])
    
    # Load data for this partition
    trainloader, testloader = load_data(partition_id, num_partitions, batch_size)
    
    # Set up Ethereum account - we'll use a temporary address for now
    # The server will assign a proper address
    temp_address = f"0x0000000000000000000000000000000000{partition_id:06d}"
    
    # Set up contract interface
    contract = PrivateShapleyContract()
    
    # Create and return the client
    return PrivateShapleyClient(
        trainloader=trainloader,
        testloader=testloader,
        epochs=epochs,
        learning_rate=learning_rate,
        trainer_address=temp_address,
        private_key="0x0000000000000000000000000000000000000000000000000000000000000000",  # Temporary private key
        contract=contract,
    ).to_client()


# Create the client app
app = ClientApp(client_fn=client_fn)