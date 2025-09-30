from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar


class DecentralShapleyValueClient(NumPyClient):
    """Client wrapper for decentralized Shapley value evaluation.

    This wrapper extends a NumPyClient to support coalition evaluation
    in decentralized Shapley value calculation. It passes through all client
    operations while adding coalition metadata to evaluation results.

    Attributes:
        client: The underlying NumPyClient being wrapped.
    """

    client: NumPyClient

    def __init__(self, client: NumPyClient) -> None:
        """Initialize the decentralized Shapley value client.

        Args:
            client: The base NumPyClient to wrap.
        """
        super().__init__()
        self.client = client

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Get model parameters from the underlying client.

        Args:
            config: Configuration dictionary.

        Returns:
            Model parameters as numpy arrays.
        """
        return self.client.get_parameters(config)

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """Get client properties from the underlying client.

        Args:
            config: Configuration dictionary.

        Returns:
            Dictionary of client properties.
        """
        return self.client.get_properties(config)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Train the model using the underlying client.

        Args:
            parameters: Model parameters as numpy arrays.
            config: Configuration dictionary for training.

        Returns:
            Tuple containing updated parameters, number of examples, and metrics.
        """
        return self.client.fit(parameters, config)

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Evaluate coalition parameters and return results with coalition ID.

        Evaluates the provided model parameters using the underlying client and
        augments the results with the coalition ID from the configuration.

        Args:
            parameters: Model parameters to evaluate as numpy arrays.
            config: Configuration dictionary containing the coalition ID.

        Returns:
            Tuple containing loss value, number of examples, and metrics dictionary
            with the coalition ID included.
        """
        loss, num_examples, metrics = self.client.evaluate(parameters, {})
        return loss, num_examples, {"id": config["id"]} | metrics
