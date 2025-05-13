"""Server application for the Private Shapley example."""

import os
import hashlib
import secrets
from typing import Dict, List, Tuple, cast, Optional, Any
import numpy as np
from dotenv import load_dotenv
from eth_account import Account
from flwr.common import Context, Metrics, Parameters, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .task import Net, get_weights, test
from .contract import PrivateShapleyContract
from .commit import CoalitionManager, address_to_bitfield

# Load environment variables
load_dotenv()

# Global coalition manager
coalition_manager = CoalitionManager()

class PrivateShapleyStrategy(FedAvg):
    """Strategy for Private Shapley federated learning."""
    
    def __init__(
        self,
        contract: PrivateShapleyContract,
        aggregator_address: str,
        private_key: str,
        **kwargs
    ):
        """Initialize the strategy.
        
        Args:
            contract: Interface to the PrivateShapley contract
            aggregator_address: Ethereum address of the aggregator
            private_key: Private key for the aggregator
            **kwargs: Additional parameters for FedAvg
        """
        super().__init__(**kwargs)
        self.contract = contract
        self.aggregator_address = aggregator_address
        self.private_key = private_key
        self.round = 0
        self.trainer_indices = {}  # address -> index
        self.registered_trainers = set()  # addresses of registered trainers
        self.client_commitments = {}  # round -> {address -> commitment}
        self.coalitions = {}  # round -> list of coalition_addresses
        self.coalition_parameters = {}  # round -> {coalition_id -> parameters}
        self.coalition_results = {}  # round -> {coalition_id -> result}
        
        # Set the owner address in the contract
        self.contract.set_owner(aggregator_address)
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize model parameters."""
        # Get initial model parameters
        initial_parameters = super().initialize_parameters(client_manager)
        
        # Reset round counter
        self.round = 0
        
        return initial_parameters
    
    
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict]]:
        """Configure the next round of training."""
        # Update round number
        self.round = server_round
        print(f"Starting round {server_round}")
        
        # Get base configuration from parent
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        
        # Enhance configuration for each client
        enhanced_instructions = []
        for client, fit_ins in client_instructions:
            # Get client ID from proxy
            client_id = client.cid
            
            # Generate a configuration for this client
            config = dict(fit_ins.config)  # Create a copy of the config dict
            
            # Add round ID
            config["round_id"] = server_round
            
            # Check if we need to register this trainer
            if client_id not in self.registered_trainers:
                try:
                    # For testing, generate an Ethereum address from the client ID
                    # In a real deployment, you'd use the actual client's Ethereum address
                    # Here we derive a deterministic address from the client ID
                    client_address = self._get_eth_address_for_client(client_id)
                    
                    # Register the trainer
                    trainer_index = self.contract.register_trainer(
                        client_address, self.private_key
                    )
                    self.trainer_indices[client_id] = trainer_index
                    self.registered_trainers.add(client_id)
                    print(f"Registered trainer {client_id} with index {trainer_index}")
                    
                    # Add trainer index to config
                    config["trainer_index"] = trainer_index
                    config["trainer_address"] = client_address
                    
                    # Register with coalition manager
                    coalition_manager.register_trainer(client_address, trainer_index)
                except Exception as e:
                    print(f"Error registering trainer {client_id}: {e}")
            elif client_id in self.trainer_indices:
                # Send the trainer index
                config["trainer_index"] = self.trainer_indices[client_id]
                client_address = self._get_eth_address_for_client(client_id)
                config["trainer_address"] = client_address
            
            # Create a new FitIns with the updated config
            from flwr.common import FitIns
            new_fit_ins = FitIns(parameters=fit_ins.parameters, config=config)
            
            # Save the enhanced configuration
            enhanced_instructions.append((client, new_fit_ins))
        
        return enhanced_instructions

    def _get_eth_address_for_client(self, client_id: str) -> str:
        """Generate a deterministic Ethereum address from a client ID.
        
        Args:
            client_id: The client ID
            
        Returns:
            An Ethereum address derived from the client ID
        """
        # Use a simple hash to derive a private key from the client ID
        import hashlib
        
        # Create a deterministic hash from the client ID
        client_id_bytes = str(client_id).encode('utf-8')
        hashed = hashlib.sha256(client_id_bytes).digest()
        
        # Use the hash to create an account
        from eth_account import Account
        account = Account.from_key(hashed)
        return account.address
    

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Any]],
        failures: List[Any],
    ) -> Tuple[Optional[Parameters], Dict[str, Any]]:
        """Aggregate model updates and create coalitions."""
        # Collect trainer addresses and commitments
        trainer_addresses = []
        commitments = {}
        
        for client, fit_res in results:
            trainer_address = fit_res.metrics.get("trainer_address")
            commitment = fit_res.metrics.get("commitment")
            
            if trainer_address and commitment:
                trainer_addresses.append(trainer_address)
                commitments[trainer_address] = bytes.fromhex(commitment)
        
        # Store commitments for this round
        self.client_commitments[server_round] = commitments
        
        # Create coalitions
        self._create_coalitions(server_round, trainer_addresses, results)
        
        # Proceed with normal aggregation
        return super().aggregate_fit(server_round, results, failures)
    
    def _create_coalitions(
        self,
        round_id: int,
        trainer_addresses: List[str],
        results: List[Tuple[ClientProxy, Any]],
    ) -> None:
        """Create coalitions and publish coalition data.
        
        Args:
            round_id: The current round ID
            trainer_addresses: List of trainer addresses
            results: Training results from clients
        """
        if not trainer_addresses:
            print("No trainers to form coalitions")
            return
        
        # For simplicity, we'll create one coalition with all trainers
        # In a real implementation, you would create multiple coalitions
        coalition_addresses = trainer_addresses
        
        # Create coalition ID (keccak256 hash of addresses)
        coalition_data = b"".join(address.encode() for address in sorted(coalition_addresses))
        coalition_id = hashlib.sha3_256(coalition_data).digest()
        
        # Store coalition for this round
        if round_id not in self.coalitions:
            self.coalitions[round_id] = []
        self.coalitions[round_id].append(coalition_addresses)
        
        # Get indices for these addresses
        trainer_indices = [self.trainer_indices.get(addr) for addr in coalition_addresses if addr in self.trainer_indices]
        if None in trainer_indices:
            print("Some trainers not registered, skipping coalition creation")
            return
        
        # Create coalition bitfield and Merkle root
        try:
            bitfield, merkle_root = coalition_manager.create_coalition_data(
                round_id, coalition_addresses
            )
            
            # Publish coalition data to the contract
            self.contract.publish_coalition_data(
                coalition_id, bitfield, merkle_root, self.private_key
            )
            print(f"Published coalition {coalition_id.hex()} with {len(coalition_addresses)} trainers")
            
            # Aggregate parameters for this coalition
            coalition_results = [res for client, res in results 
                               if res.metrics.get("trainer_address") in coalition_addresses]
            
            # Simple weighted averaging
            weights = [res.num_examples for res in coalition_results]
            total_weight = sum(weights)
            
            weighted_params = []
            for i in range(len(coalition_results[0].parameters.tensors)):
                # Get the i-th parameter from each client
                params_i = [res.parameters.tensors[i] * (w / total_weight) 
                          for res, w in zip(coalition_results, weights)]
                # Sum them up
                weighted_params.append(sum(params_i))
            
            # Store coalition parameters
            if round_id not in self.coalition_parameters:
                self.coalition_parameters[round_id] = {}
            self.coalition_parameters[round_id][coalition_id] = ndarrays_to_parameters(weighted_params)
            
        except Exception as e:
            print(f"Error creating coalition: {e}")
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, Dict]]:
        """Configure the evaluation phase."""
        # Get base configuration from parent
        client_instructions = super().configure_evaluate(
            server_round, parameters, client_manager
        )
        
        # Check if we have any coalitions for testing
        coalitions_to_test = []
        if server_round in self.coalition_parameters:
            coalitions_to_test = list(self.coalition_parameters[server_round].items())
        
        if not coalitions_to_test:
            print("No coalitions to test in this round")
            return client_instructions
        
        # Choose random clients to test each coalition
        enhanced_instructions = []
        client_proxies = [client for client, _ in client_instructions]
        
        for i, (coalition_id, coalition_params) in enumerate(coalitions_to_test):
            # Pick a client to test this coalition
            client_idx = i % len(client_proxies)
            client = client_proxies[client_idx]
            
            # Get the original instructions
            eval_ins = client_instructions[client_idx][1]
            
            # Create enhanced config
            config = eval_ins.config.copy()
            config["coalition_id"] = coalition_id.hex()
            
            # Add this to the enhanced instructions
            enhanced_instructions.append((client, eval_ins._replace(
                parameters=coalition_params,
                config=config
            )))
            
            # Publish the result ourselves (in a real system this would be done by the tester)
            self._test_and_publish_result(coalition_id, coalition_params, server_round)
        
        return enhanced_instructions
    
    def _test_and_publish_result(
        self,
        coalition_id: bytes,
        parameters: Parameters,
        round_id: int,
    ) -> None:
        """Test a coalition and publish the result.
        
        Args:
            coalition_id: The coalition ID
            parameters: The coalition's model parameters
            round_id: The current round ID
        """
        try:
            # Create a model for testing
            net = Net()
            weights = parameters.tensors
            
            # Convert to PyTorch tensors
            import torch
            from collections import OrderedDict
            net_state_dict = OrderedDict()
            for i, (name, _) in enumerate(net.state_dict().items()):
                net_state_dict[name] = torch.tensor(weights[i])
            
            net.load_state_dict(net_state_dict)
            
            # Simple test - in reality, this would be done on a test dataset
            # Here we're just using a random score for demonstration
            score = secrets.randbelow(1000) / 1000.0  # Random score between 0 and 1
            
            # Scale up for precision (multiply by 10^6)
            result = int(score * 1_000_000)
            
            # Publish result to the contract
            self.contract.publish_result(
                coalition_id, result, self.aggregator_address, self.private_key
            )
            print(f"Published result {score:.4f} for coalition {coalition_id.hex()}")
            
            # Store result
            if round_id not in self.coalition_results:
                self.coalition_results[round_id] = {}
            self.coalition_results[round_id][coalition_id] = score
            
        except Exception as e:
            print(f"Error testing and publishing result: {e}")
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, Any]],
        failures: List[Any],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Aggregate evaluation results."""
        # Just use the parent implementation for now
        return super().aggregate_evaluate(server_round, results, failures)


def server_fn(context: Context) -> ServerAppComponents:
    """Create and configure the server components.
    
    Args:
        context: The server context provided by Flower
        
    Returns:
        Configured server components
    """
    # Load environment variables
    load_dotenv()
    
    # Parse configuration
    server_rounds = int(context.run_config["num-server-rounds"])
    fraction_fit = float(context.run_config["fraction-fit"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    min_available_clients = int(context.run_config["min-available-clients"])
    
    # Initialize model parameters
    net = Net()
    initial_parameters = ndarrays_to_parameters(get_weights(net))
    
    # # Set up Ethereum account for the aggregator
    # mnemonic = os.getenv("PRIVATE_SHAPLEY_MNEMONIC")
    # if not mnemonic:
    #     raise ValueError("PRIVATE_SHAPLEY_MNEMONIC not found in environment variables")
    
    # Enable HD wallet features
    Account.enable_unaudited_hdwallet_features()
    
    # TODO: Add aggregator address as owner or use the first account as aggregator
    
    # Create an account for the aggregator (using index 0)
    # hd_path = "m/44'/60'/0'/0/0"
    # account = Account.from_mnemonic(mnemonic, account_path=hd_path)
    aggregator_address = os.getenv("PRIVATE_OWNER_ADDR")
    
    private_key = os.getenv("PRIVATE_OWNER_KEY")
    
    print(f"Aggregator address: {aggregator_address}")
    
    # Set up contract interface
    contract = PrivateShapleyContract()
    
    # Create the strategy
    strategy = PrivateShapleyStrategy(
        contract=contract,
        aggregator_address=aggregator_address,
        private_key=private_key,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters,
    )
    
    # Create server configuration
    server_config = ServerConfig(num_rounds=server_rounds)
    
    # Return the server components
    return ServerAppComponents(strategy=strategy, config=server_config)


# Create the server app
app = ServerApp(server_fn=server_fn)