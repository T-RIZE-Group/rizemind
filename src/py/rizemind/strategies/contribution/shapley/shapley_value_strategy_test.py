import uuid
from math import isclose

import pytest
from flwr.common import EvaluateIns, EvaluateRes, FitRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from rizemind.compensation.shapley.shapley_value_strategy import (
    Coalition,
    ShapleyValueStrategy,
)


class MockShapleyValueStrategy(ShapleyValueStrategy):
    def __init__(self, strategy, model):
        ShapleyValueStrategy.__init__(self, strategy, model)
        self.last_round_parameters = Parameters([], "")

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, bool | bytes | float | int | str]]:
        return None, {}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return []

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        return None


class DummyFitRes:
    def __init__(self, trainer_address: str):
        self.metrics = {"trainer_address": trainer_address}

    def __eq__(self, other):
        return isinstance(other, DummyFitRes) and self.metrics == other.metrics

    def __repr__(self):
        return f"DummyFitRes({self.metrics['trainer_address']})"


class DummyClientProxy:
    def __init__(self, identifier: str):
        self.identifier = identifier

    def __eq__(self, other):
        return (
            isinstance(other, DummyClientProxy) and self.identifier == other.identifier
        )

    def __repr__(self):
        return f"DummyClientProxy({self.identifier})"


class DummyStrategy:
    def __init__(self):
        return

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes]],
    ):
        return Parameters([], ""), {}


@pytest.fixture
def mocked_shapley_value_strategy():
    return MockShapleyValueStrategy(
        DummyStrategy(),  # type: ignore
        "dummy_model",  # type: ignore
    )


# -------------------------------
# Test for coalition creation
# -------------------------------


def sort_key(coalition):
    key_tuples = [(member, member) for member in sorted(coalition.members)]
    return (len(coalition.members), key_tuples)


@pytest.mark.parametrize(
    "results, expected_members_lists",
    [
        # Test case with one tuple:
        (
            [(DummyClientProxy("1"), DummyFitRes("1"))],
            [
                [],  # empty coalition
                ["1"],  # coalition with trainer "1"
            ],
        ),
        # Test case with two tuples:
        (
            [
                (DummyClientProxy("1"), DummyFitRes("1")),
                (DummyClientProxy("2"), DummyFitRes("2")),
            ],
            [
                [],  # empty coalition
                ["1"],  # single coalition for trainer "1"
                ["2"],  # single coalition for trainer "2"
                ["1", "2"],  # coalition with both trainers
            ],
        ),
        # Test case with three tuples:
        (
            [
                (DummyClientProxy("1"), DummyFitRes("1")),
                (DummyClientProxy("2"), DummyFitRes("2")),
                (DummyClientProxy("3"), DummyFitRes("3")),
            ],
            [
                [],  # empty coalition
                ["1"],
                ["2"],
                ["3"],
                ["1", "2"],
                ["1", "3"],
                ["2", "3"],
                ["1", "2", "3"],
            ],
        ),
    ],
)
def test_create_coalitions(
    mocked_shapley_value_strategy, results, expected_members_lists
):
    res = mocked_shapley_value_strategy.create_coalitions(0, results)
    # Extract and sort the members list from each Coalition.
    result_members = sorted([sorted(coalition.members) for coalition in res])
    expected_sorted = sorted([sorted(members) for members in expected_members_lists])
    assert result_members == expected_sorted, (
        f"Expected {expected_sorted}, got {result_members}"
    )


# -------------------------------
# Test for contribution computation
# -------------------------------


def generate_compute_contribution_params():
    test_cases = [
        {
            "cs": [
                ([], 90),
                (["address_1"], 92),
                (["address_2"], 93),
                (["address_1", "address_2"], 90),
            ],
            "player_x_outcome": [("address_1", -0.5), ("address_2", 0.5)],
        },
        {
            "cs": [
                ([], 0),
                (["address_1"], 0),
                (["address_2"], 0),
                (["address_1", "address_2"], 1500),
            ],
            "player_x_outcome": [("address_1", 750), ("address_2", 750)],
        },
        {
            "cs": [
                ([], 0),
                (["address_1"], 1000),
                (["address_2"], 1000),
                (["address_1", "address_2"], 2000),
            ],
            "player_x_outcome": [("address_1", 1000), ("address_2", 1000)],
        },
        {
            "cs": [
                ([], 0),
                (["address_1"], 0),
                (["address_2"], 0),
                (["address_1", "address_2"], 100),
            ],
            "player_x_outcome": [("address_1", 50), ("address_2", 50)],
        },
        {
            "cs": [
                ([], 0),
                (["address_1"], 100),
                (["address_2"], 0),
                (["address_1", "address_2"], 100),
            ],
            "player_x_outcome": [("address_1", 100), ("address_2", 0)],
        },
        {
            "cs": [
                ([], 0),
                (["address_1"], 1000000),
                (["address_2"], 200000),
                (["address_1", "address_2"], 1400000),
            ],
            "player_x_outcome": [("address_1", 1100000), ("address_2", 300000)],
        },
        {
            "cs": [
                ([], 0),
                (["address_1"], 1000000),
                (["address_2"], 500000),
                (["address_1", "address_2"], 1250000),
            ],
            "player_x_outcome": [("address_1", 875000), ("address_2", 375000)],
        },
        {
            "cs": [
                ([], 0),
                (["address_1"], 3600000),
                (["address_2"], 600000),
                (["address_1", "address_2"], 3500000),
            ],
            "player_x_outcome": [("address_1", 3250000), ("address_2", 250000)],
        },
        {
            "cs": [
                ([], 0),
                (["address_1"], 3600000),
                (["address_2"], 600000),
                (["address_1", "address_2"], 3500000),
            ],
            "player_x_outcome": [("address_1", 3250000), ("address_2", 250000)],
        },
        {
            "cs": [
                (
                    # Checks if the i-th bit is set to 1 in the given coalition
                    # It will create a "numerical address" for it using 1 << i
                    [1 << i for i in range(7) if (coalition >> i) & 1],
                    0 if coalition == 0 else -100,
                )
                for coalition in range(2**7)
            ],
            "player_x_outcome": [
                ("0b1", -14.285714285714),
                ("0b10", -14.285714285714),
                ("0b100", -14.285714285714),
                ("0b1000", -14.285714285714),
                ("0b10000", -14.285714285714),
                ("0b100000", -14.285714285714),
                ("0b1000000", -14.285714285714),
            ],
        },
        {
            "cs": [
                (
                    [1 << i for i in range(7) if (coalition >> i) & 1],
                    3000000
                    if coalition == 0b1111110
                    else 3500000
                    if coalition == 0b1111111
                    else 0,
                )
                for coalition in range(2**7)
            ],
            "player_x_outcome": [
                ("0b1", 71428.571428571),
                ("0b10", 571428.57142857),
                ("0b100", 571428.57142857),
                ("0b1000", 571428.57142857),
                ("0b10000", 571428.57142857),
                ("0b100000", 571428.57142857),
                ("0b1000000", 571428.57142857),
            ],
        },
    ]

    for case in test_cases:
        cs: list[Coalition] = []
        for addresses, score in case["cs"]:
            addresses = [
                bin(address) if isinstance(address, int) else address
                for address in addresses
            ]
            id = uuid.uuid4()
            id = str(id)
            coalition = Coalition(id, addresses, Parameters([], ""), {})  # type: ignore
            coalition.loss = score
            cs.append(coalition)
        for player, expected in case["player_x_outcome"]:
            yield pytest.param(cs, player, expected)


@pytest.mark.parametrize(
    "cs, player, expected", list(generate_compute_contribution_params())
)
def test_compute_contributions(mocked_shapley_value_strategy, cs, player, expected):
    # Call the function with the coalition and score list.
    computed = mocked_shapley_value_strategy.compute_contributions(cs)
    # Convert the result list to a dictionary for easier lookup.
    computed_dict = {addr: value for addr, value in computed}
    # Assert that the computed contribution for the given player is as expected.
    assert isclose(computed_dict[player], expected)
