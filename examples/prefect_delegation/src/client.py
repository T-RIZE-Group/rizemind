"""Trainer node — delegates training to Prefect instead of training in-process.

This is the "small node within the trainer's infrastructure". It holds the swarm
identity (Ethereum key) and speaks to the aggregator, but it does NOT train. On
``fit`` it:

1. publishes the global weights to the (scoped) weights-in bucket,
2. triggers the delegated-training Prefect flow on the trainer's ETL/compute plane,
3. waits for the flow to complete,
4. reads the produced weights back from the (scoped) weights-out bucket,
5. returns them to the aggregator, where Rizemind's notary mod signs the update.

The dataset is assumed already exported by the trainer's ETL to the dataset bucket.
For a runnable demo the client seeds a synthetic export on first use (emulating the
ETL); delete that block when wiring a real ETL.
"""

from __future__ import annotations

from eth_account import Account
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from rizemind.authentication import authentication_mod
from rizemind.authentication.config import ACCOUNT_CONFIG_STATE_KEY, AccountConfig
from rizemind.authentication.notary.model.mod import model_notary_mod
from rizemind.configuration.toml_config import TomlConfig
from rizemind.strategies.contribution.shapley.decentralized.shapley_value_client import (
    DecentralShapleyValueClient,
)
from rizemind.web3 import Web3Config
from rizemind.web3.config import WEB3_CONFIG_STATE_KEY

from .flow import delegated_training
from .storage import make_store
from .task import make_synthetic_export, weights_from_bytes, weights_to_bytes


class DelegatingClient(NumPyClient):
    """A Flower client that offloads training to a Prefect flow."""

    def __init__(
        self,
        *,
        partition_id: int,
        store_backend: str,
        store_location: str,
        epochs: int,
        lr: float,
        mlflow_uri: str,
        mlflow_experiment: str,
        approved_image_digest: str | None,
    ) -> None:
        self.partition_id = partition_id
        self.store = make_store(store_backend, store_location)
        self.store_backend = store_backend
        self.store_location = store_location
        self.epochs = epochs
        self.lr = lr
        self.mlflow_uri = mlflow_uri
        self.mlflow_experiment = mlflow_experiment
        self.approved_image_digest = approved_image_digest

        # --- demo only: emulate the trainer's ETL having exported a dataset ---
        self.dataset_uri = f"trainer-{partition_id}/dataset.npz"
        if not self.store.exists(self.dataset_uri):
            self.store.put_bytes(
                self.dataset_uri, make_synthetic_export(seed=partition_id + 1)
            )
        # ----------------------------------------------------------------------

    def _round_uri(self, round_id: int, name: str) -> str:
        return f"trainer-{self.partition_id}/round-{round_id}/{name}"

    def fit(self, parameters, config):
        round_id = int(config.get("current_round", config.get("round", 0)))
        weights_in_uri = self._round_uri(round_id, "weights_in.npz")
        weights_out_uri = self._round_uri(round_id, "weights_out.npz")

        # 1. publish global weights where only the training step can read them
        self.store.put_bytes(weights_in_uri, weights_to_bytes(parameters))

        # 2/3. trigger the flow on the trainer's infra and wait for completion.
        # Locally we invoke the flow in-process; in production replace this with
        #   from prefect.deployments import run_deployment
        #   run = run_deployment(name="delegated-training/trainer-etl", parameters={...})
        # which submits to the trainer's Prefect work pool and blocks until done.
        produced_uri = delegated_training(
            dataset_uri=self.dataset_uri,
            weights_in_uri=weights_in_uri,
            weights_out_uri=weights_out_uri,
            round_id=round_id,
            epochs=self.epochs,
            lr=self.lr,
            approved_digest=self.approved_image_digest,
            store_backend=self.store_backend,
            store_location=self.store_location,
            mlflow_uri=self.mlflow_uri,
            mlflow_experiment=self.mlflow_experiment,
        )

        # 4. read the produced weights back and hand them to the aggregator
        updated = weights_from_bytes(self.store.get_bytes(produced_uri))
        num_examples = 512  # reported by the ETL export in production
        return updated, num_examples, {"delegated": True, "artifact_uri": produced_uri}

    def evaluate(self, parameters, config):
        # Evaluation stays lightweight and local on the small node.
        import torch

        from .task import Net, load_partition, set_weights

        net = Net()
        set_weights(net, parameters)
        x_train, y_train, x_val, y_val = load_partition(
            self.store.get_bytes(self.dataset_uri)
        )
        net.eval()
        with torch.no_grad():
            logits = net(x_val)
            loss = float(torch.nn.functional.cross_entropy(logits, y_val).item())
            accuracy = float((logits.argmax(dim=1) == y_val).float().mean().item())
        return loss, len(x_val), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])

    config = TomlConfig("./pyproject.toml")
    account_config = AccountConfig(
        **config.get("tool.eth.account") | {"default_account_index": partition_id + 1}
    )
    context.state.config_records[ACCOUNT_CONFIG_STATE_KEY] = (
        account_config.to_config_record()
    )
    web3_config = Web3Config(**config.get("tool.web3"))
    context.state.config_records[WEB3_CONFIG_STATE_KEY] = web3_config.to_config_record()

    delegation = config.get("tool.delegation")
    client = DelegatingClient(
        partition_id=partition_id,
        store_backend=delegation["store_backend"],
        store_location=delegation["store_location"],
        epochs=int(delegation["epochs"]),
        lr=float(delegation["lr"]),
        mlflow_uri=delegation.get("mlflow_uri", ""),
        mlflow_experiment=delegation.get("mlflow_experiment", "prefect-delegation"),
        approved_image_digest=delegation.get("approved_image_digest") or None,
    )
    shapley_client = DecentralShapleyValueClient(client=client)
    return shapley_client.to_client()


Account.enable_unaudited_hdwallet_features()
app = ClientApp(client_fn, mods=[authentication_mod, model_notary_mod])
