import math
import statistics
from typing import Any

import pytest
from flwr.common import Code, EvaluateRes, Parameters, Status
from rizemind.strategies.contribution.calculators.calculator import TrainerSetAggregate


@pytest.fixture
def trainer_set_params() -> dict[str, Any]:
    return {
        "id": "set_1",
        "members": ["0x1234", "0x1235"],
        "parameters": Parameters(tensors=[], tensor_type="torch"),
        "config": {"c1": "v1"},
    }


@pytest.fixture
def empty_trainer_set_aggregate(
    trainer_set_params: dict[str, Any],
) -> TrainerSetAggregate:
    return TrainerSetAggregate(**trainer_set_params)


@pytest.fixture
def properly_populated_trainer_set_aggregate(
    empty_trainer_set_aggregate: TrainerSetAggregate,
) -> TrainerSetAggregate:
    res1 = EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=0.7967,
        num_examples=1200,
        metrics={"r2": 0.83},
    )
    res2 = EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=0.4562,
        num_examples=1200,
        metrics={"r2": 0.91},
    )
    res3 = EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=0.9124,
        num_examples=1200,
        metrics={"r2": 0.63},
    )
    empty_trainer_set_aggregate.insert_res(res1)
    empty_trainer_set_aggregate.insert_res(res2)
    empty_trainer_set_aggregate.insert_res(res3)

    return empty_trainer_set_aggregate


@pytest.fixture
def improperly_populated_trainer_set_aggregate(
    empty_trainer_set_aggregate: TrainerSetAggregate,
) -> TrainerSetAggregate:
    res1 = EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=0.7967,
        num_examples=1200,
        metrics={"r2": 0.83},
    )
    res2 = EvaluateRes(
        status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="err"),
        loss=float("inf"),
        num_examples=1200,
        metrics={"r2": -1 * float("inf")},
    )
    res3 = EvaluateRes(
        status=Status(code=Code.GET_PARAMETERS_NOT_IMPLEMENTED, message="err"),
        loss=float("inf"),
        num_examples=1200,
        metrics={},
    )
    empty_trainer_set_aggregate.insert_res(res1)
    empty_trainer_set_aggregate.insert_res(res2)
    empty_trainer_set_aggregate.insert_res(res3)

    return empty_trainer_set_aggregate


def test_initialization(
    empty_trainer_set_aggregate: TrainerSetAggregate,
    trainer_set_params: dict[str, Any],
):
    tsa = empty_trainer_set_aggregate
    params = trainer_set_params

    assert tsa.id == params["id"]
    assert tsa.members == params["members"]
    assert tsa.parameters == params["parameters"]
    assert tsa.config == params["config"]
    assert tsa._evaluation_res == []


def test_insert_res(empty_trainer_set_aggregate: TrainerSetAggregate):
    tsa = empty_trainer_set_aggregate
    assert len(tsa._evaluation_res) == 0
    eval_res = EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=0.5,
        num_examples=100,
        metrics={},
    )

    tsa.insert_res(eval_res)

    assert len(tsa._evaluation_res) == 1
    assert tsa._evaluation_res[0] == eval_res


def test_get_loss_empty(empty_trainer_set_aggregate: TrainerSetAggregate):
    tsa = empty_trainer_set_aggregate
    assert tsa.get_loss() == float("Inf")


def test_get_loss_properly_populated(
    properly_populated_trainer_set_aggregate: TrainerSetAggregate,
):
    tsa = properly_populated_trainer_set_aggregate
    expected_mean = statistics.mean([0.7967, 0.4562, 0.9124])
    assert tsa.get_loss() == pytest.approx(expected_mean)


def test_get_loss_with_custom_aggregator(
    properly_populated_trainer_set_aggregate: TrainerSetAggregate,
):
    """Tests using a custom aggregator (max) for loss."""
    tsa = properly_populated_trainer_set_aggregate
    expected_max = max([0.7967, 0.4562, 0.9124])
    assert tsa.get_loss(aggregator=max) == expected_max


def test_get_loss_on_empty_returns_inf(
    empty_trainer_set_aggregate: TrainerSetAggregate,
):
    tsa = empty_trainer_set_aggregate
    assert math.isinf(tsa.get_loss())


def test_get_loss_with_inf(
    improperly_populated_trainer_set_aggregate: TrainerSetAggregate,
):
    tsa = improperly_populated_trainer_set_aggregate
    assert tsa.get_loss() == float("Inf")


def test_get_metric_empty(empty_trainer_set_aggregate: TrainerSetAggregate):
    tsa = empty_trainer_set_aggregate
    default_value = 0.0
    assert (
        tsa.get_metric("r2", default=default_value, aggregator=statistics.mean)
        == default_value
    )


def test_get_metric_properly_populated_with_custom_aggregator(
    properly_populated_trainer_set_aggregate: TrainerSetAggregate,
):
    """Tests metric aggregation with a custom aggregator (mean)."""
    tsa = properly_populated_trainer_set_aggregate
    expected_mean = statistics.mean([0.83, 0.91, 0.63])
    metric_values = tsa.get_metric("r2", default=0.0, aggregator=statistics.mean)
    assert metric_values == pytest.approx(expected_mean)


def test_get_metric_nonexistent(
    properly_populated_trainer_set_aggregate: TrainerSetAggregate,
):
    """Tests getting a metric that does not exist in the results."""
    tsa = properly_populated_trainer_set_aggregate
    default_value = -1.0
    assert (
        tsa.get_metric("accuracy", default=default_value, aggregator=statistics.mean)
        == -1.0
    )


def test_get_metric_partially_present(
    improperly_populated_trainer_set_aggregate: TrainerSetAggregate,
):
    tsa = improperly_populated_trainer_set_aggregate

    assert tsa.get_metric("r2", default=0.0, aggregator=min) == 0.0
