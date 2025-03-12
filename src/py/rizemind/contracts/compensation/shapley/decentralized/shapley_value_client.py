from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar


class DecentralShapleyValueClient(NumPyClient):
    client: NumPyClient

    def __init__(self, client: NumPyClient) -> None:
        super().__init__()
        self.client = client

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return self.client.get_parameters(config)

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        return self.client.get_properties(config)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        return self.client.fit(parameters, config)

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        loss, num_examples, metrics = self.client.evaluate(parameters, {})
        return loss, num_examples, {"id": config["id"]} | metrics
