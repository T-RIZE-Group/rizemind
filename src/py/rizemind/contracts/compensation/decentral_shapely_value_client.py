import json
from typing import cast
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from flwr.common.typing import Parameters
from flwr.common.parameter import parameters_to_ndarrays as flwr_parameters_to_ndarrays
from rizemind.contracts.compensation.decentral_util import decode_parameters


class DecentralShapelyValueClient(NumPyClient):
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

    # TODO: we could use the id to get the parameters of the model in strategy
    # instead of sending them all from here
    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        evaluated_json = dict()
        print("config shape")
        with open("/home/mikaeil/log.txt", "w") as f:
            print(config, file=f)
        evaluation_json: dict = json.loads(cast(str, config["evaluation_json"]))
        for id, coalition_parameters in evaluation_json.values():
            # coalition_parameters = Parameters(coalition_parameters_dict['tensors'], coalition_parameters_dict['tensor_type'])
            coalition_parameters = decode_parameters(coalition_parameters)
            coalition_ndarrays = flwr_parameters_to_ndarrays(coalition_parameters)
            loss, num_examples, metrics = self.client.evaluate(coalition_ndarrays, {})
            evaluated_json[id] = {
                "loss": loss,
                "num_examples": num_examples,
                "accuracy": metrics["accuracy"],
                "parameters": coalition_parameters.__dict__,
            }
        return 0, 0, {"evaluated_json": json.dumps(evaluated_json)}
