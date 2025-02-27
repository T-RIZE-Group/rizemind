from eth_typing import Address
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from rizemind.contracts.compensation.shapely_value_strategy import ShapelyValueStrategy


class MockShapelyValueStrategy(ShapelyValueStrategy):
    def __init__(self, strategy, model):
        ShapelyValueStrategy.__init__(self, strategy, model)

    def calculate(self, client_ids: list[Address]) -> tuple[list[Address], list[int]]:
        return super().calculate(client_ids)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        return super().aggregate_fit(server_round, results, failures)

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return super().initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, bool | bytes | float | int | str]]:
        return super().aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        return super().evaluate(server_round, parameters)

    def evaluate_coalitions(self):
        return None


class DummyFitRes:
    def __init__(self, trainer_address: str):
        self.metrics = {"trainer_address": trainer_address}


def test_create_coalitions():
    CREATE_COALITION_TEST_CASES = [
        {"fit_res": [DummyFitRes("1")], "coalitions": [[], ["1"]]},
        {
            "fit_res": [DummyFitRes("1"), DummyFitRes("2")],
            "coalitions": [[], ["1"], ["2"], ["1", "2"]],
        },
        {
            "fit_res": [DummyFitRes("1"), DummyFitRes("2"), DummyFitRes("3")],
            "coalitions": [
                [],
                ["1"],
                ["2"],
                ["3"],
                ["1", "2"],
                ["1", "3"],
                ["2", "3"],
                ["1", "2", "3"],
            ],
        },
    ]
    mocked_shapely_value_strategy = MockShapelyValueStrategy(
        "dummy_strategy",  # type: ignore
        "dummy_model",  # type: ignore
    )
    for test_case in CREATE_COALITION_TEST_CASES:
        res = mocked_shapely_value_strategy.create_coalitions(test_case["fit_res"])
        result_sorted = sorted(res, key=lambda lst: (len(lst), lst))
        expected_sorted = sorted(
            test_case["coalitions"], key=lambda lst: (len(lst), lst)
        )
        assert result_sorted == expected_sorted, (
            f"Expected {expected_sorted}, got {result_sorted}"
        )


def test_compute_contributions():
    COMPUTE_CONTRIBUTION_TEST_CASES = [
        {
            "cs": [(0b00, 90), (0b01, 92), (0b10, 93), (0b11, 90)],
            "player_x_outcome": ([0b01, -0.5], [0b10, 0.5]),
        },
        {
            "cs": [(0b00, 0), (0b01, 0), (0b10, 0), (0b11, 1500)],
            "player_x_outcome": ([0b01, 750], [0b10, 750]),
        },
        {
            "cs": [(0b00, 0), (0b01, 1000), (0b10, 1000), (0b11, 2000)],
            "player_x_outcome": ([0b01, 1000], [0b10, 1000]),
        },
        {
            "cs": [(0b00, 0), (0b01, 0), (0b10, 0), (0b11, 100)],
            "player_x_outcome": ([0b01, 50], [0b10, 50]),
        },
        {
            "cs": [(0b00, 0), (0b01, 100), (0b10, 0), (0b11, 100)],
            "player_x_outcome": ([0b01, 100], [0b10, 0]),
        },
        {
            "cs": [(0b00, 0), (0b01, 1000000), (0b10, 200000), (0b11, 1400000)],
            "player_x_outcome": ([0b01, 1100000], [0b10, 300000]),
        },
        {
            "cs": [(0b00, 0), (0b01, 1000000), (0b10, 500000), (0b11, 1250000)],
            "player_x_outcome": ([0b01, 875000], [0b10, 375000]),
        },
        {
            "cs": [(i, 0 if i == 0 else -100) for i in range(2**7)],
            "player_x_outcome": (
                [0b0000001, -14.285714285714],
                [0b0000010, -14.285714285714],
                [0b0000100, -14.285714285714],
                [0b0001000, -14.285714285714],
                [0b0010000, -14.285714285714],
                [0b0100000, -14.285714285714],
                [0b1000000, -14.285714285714],
            ),
        },
        {
            "cs": [(0b00, 0), (0b01, 3600000), (0b10, 600000), (0b11, 3500000)],
            "player_x_outcome": ([0b01, 3250000], [0b10, 250000]),
        },
        {
            "cs": [(0b00, 0), (0b01, 3600000), (0b10, 600000), (0b11, 3500000)],
            "player_x_outcome": ([0b01, 3250000], [0b10, 250000]),
        },
        {
            "cs": [
                (
                    coalition,
                    3000000
                    if coalition == 0b1111110
                    else 3500000
                    if coalition == 0b1111111
                    else 0,
                )
                for coalition in range(2**7)
            ],
            "player_x_outcome": (
                [0b0000001, 71428.571428571],  # NGO (player 1)
                [0b0000010, 571428.57142857],  # Gov subagency (player 2)
                [0b0000100, 571428.57142857],  # Gov subagency (player 3)
                [0b0001000, 571428.57142857],  # Gov subagency (player 4)
                [0b0010000, 571428.57142857],  # Gov subagency (player 5)
                [0b0100000, 571428.57142857],  # Gov subagency (player 6)
                [0b1000000, 571428.57142857],  # Gov subagency (player 7)
            ),
        },
    ]
    mocked_shapely_value_strategy = MockShapelyValueStrategy(
        "dummy_strategy",  # type: ignore
        "dummy_model",  # type: ignore
    )
    for test_case in COMPUTE_CONTRIBUTION_TEST_CASES:
        cs = test_case["cs"]
        for player_x_outcome in test_case["player_x_outcome"]:
            player = player_x_outcome[0]
            expected_contribution = player_x_outcome[1]
            calculated_contribution = (
                mocked_shapely_value_strategy.compute_contributions(player, cs)
            )
            assert round(expected_contribution, 6) == round(calculated_contribution, 6)
