from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from dp_utils import compute_model_delta, clip_update, add_gaussian_noise, apply_noisy_delta
from eval_utils import evaluate_task_metrics
from attacker import extract_attack_features
from models import LogisticRegression
import torch

class CoalitionDPStrategy(FedAvg):
    def __init__(
        self,
        coalition_schedule,
        target_client,
        dp_sigma,
        clip_norm,
        val_loader,
        probe_loader,
        global_model,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.coalition_schedule = coalition_schedule
        self.target_client = target_client
        self.dp_sigma = dp_sigma
        self.clip_norm = clip_norm
        self.round_records = []
        self.val_loader = val_loader
        self.probe_loader = probe_loader
        self.global_model = global_model
        
        # We need to maintain the server's clean parameters to compute delta
        # If parameters aren't provided in kwargs, we can extract from initial global model
        self.current_weights = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function can vary depending on the round
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Get the coalition for this round (1-indexed since server_round starts at 1)
        coalition_ids = self.coalition_schedule[server_round - 1]
        
        # Sample specific clients by matching IDs 
        # (Assuming client node_ids or string representation match)
        all_clients = client_manager.all()
        selected_clients = []
        coalition_ids_str = [str(c) for c in coalition_ids]
        for cid, client in all_clients.items():
            if str(client.cid) in coalition_ids_str or str(cid) in coalition_ids_str:
                selected_clients.append(client)
                
        if not selected_clients:
            print(f"DEBUG configure_fit failing! all_clients: {list(all_clients.keys())}")
            print(f"DEBUG condition: searching for {coalition_ids_str}")
                
        # Return client/config pairs
        return [(client, fit_ins) for client in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        # Superclass aggregation
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

            if self.clip_norm <= 0:
                if self.dp_sigma != 0:
                    raise ValueError("clip_norm must be > 0 when sigma is non-zero.")
                noisy_ndarrays = aggregated_ndarrays
            else:
                # 1. Compute delta
                delta = compute_model_delta(self.current_weights, aggregated_ndarrays)

                # 2. Clip update
                clipped_delta = clip_update(delta, self.clip_norm)

                # 3. Add Gaussian noise
                noisy_delta = add_gaussian_noise(clipped_delta, self.dp_sigma, self.clip_norm)

                # 4. Apply noisy delta
                noisy_ndarrays = apply_noisy_delta(self.current_weights, noisy_delta)
            
            # Update global state for next round 
            # Note: Do we update with noisy or clean for standard FL? 
            # Usually update with noisy so clients start from DP state.
            self.current_weights = noisy_ndarrays
            noisy_parameters = ndarrays_to_parameters(noisy_ndarrays)
            
            # Setup model with current parameters to compute metrics
            params_dict = zip(self.global_model.state_dict().keys(), noisy_ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            self.global_model.load_state_dict(state_dict, strict=True)
            
            # Task AUC
            task_metrics = evaluate_task_metrics(self.global_model, self.val_loader)
            task_auc = task_metrics["auc"]
            task_accuracy = task_metrics["accuracy"]
            
            # Attacker features
            attack_features = extract_attack_features(self.global_model, self.probe_loader)
            
            coalition_ids = self.coalition_schedule[server_round - 1]
            target_present = 1 if self.target_client in coalition_ids else 0
            
            record = {
                "round": server_round,
                "coalition": coalition_ids,
                "target_present": target_present,
                "clip_norm": self.clip_norm,
                "sigma": self.dp_sigma,
                "task_accuracy": task_accuracy,
                "task_auc": task_auc,
                "attack_features": attack_features
            }
            self.round_records.append(record)
            
            return noisy_parameters, metrics
            
        return aggregated_parameters, metrics
