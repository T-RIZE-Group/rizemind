from typing import Dict, List, Optional
from logging import INFO
import xgboost as xgb
from flwr.common.logger import log
from flwr.common import Parameters, Scalar
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from utils import BST_PARAMS
from sklearn.metrics import r2_score
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from collections import OrderedDict


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    print('--------------eval metrics server------------')
    print(eval_metrics)
    mse_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    
    r2_aggregated = (
        sum([metrics["r2-client"] * num for num, metrics in eval_metrics]) / total_num
    )
    # mse_aggregated = (
    #     sum([metrics["MSE"] * num for num, metrics in eval_metrics]) / total_num
    # )
    metrics_aggregated = {"mse": mse_aggregated, "r2": r2_aggregated}
    return metrics_aggregated

class LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to set the model parameters
def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# Evaluation function for LSTM
def get_evaluate_fn(test_data, input_size, hidden_size, num_layers):
    def evaluate_fn(server_round, parameters, config):
        if server_round == 0:
            return 0, {}
        else:
            model = LSTMNet(input_size, hidden_size, num_layers)
            set_parameters(model, parameters)
            model.to(DEVICE)
            model.eval()

            criterion = nn.MSELoss()
            loss = 0.0
            all_labels = []
            all_outputs = []

            with torch.no_grad():
                for inputs, labels in test_data:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(inputs)
                    loss += criterion(outputs, labels).item()
                    all_labels.extend(labels.cpu().numpy())
                    all_outputs.extend(outputs.cpu().numpy())

            avg_loss = loss / len(test_data.dataset)
            r2 = r2_score(all_labels, all_outputs)
            mse = mean_squared_error(all_labels, all_outputs)

            # Save the global model
            torch.save(model.state_dict(), f'/home/iman/projects/kara/Projects/T-Rize/Federated_models_reg/LSTM/global_model/global_model_round_{server_round}.pth')
            log(INFO, f"Global model saved to global_model_round_{server_round}.pth")

            return mse, {"r2": r2}

    return evaluate_fn

class CyclicClientManager(SimpleClientManager):
    """Provides a cyclic client selection rule."""

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""

        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # Return all available clients
        return [self.clients[cid] for cid in available_cids]
