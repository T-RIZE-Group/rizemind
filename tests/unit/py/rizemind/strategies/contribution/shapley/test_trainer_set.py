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
def populated_trainer_set_aggregate(
    empty_trainer_set_aggregate: TrainerSetAggregate,
) -> TrainerSetAggregate:
    res1 = EvaluateRes(
        status=Status(code=Code.OK, message="ok"),
        loss=0.7967,
        num_examples=1200,
        metrics={"r2": 0.83},
    )
    res2 = EvaluateRes(
        status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="err"),
        loss=float("inf"),
        num_examples=1200,
        metrics={},
    )
    empty_trainer_set_aggregate.insert_res(res1)
    empty_trainer_set_aggregate.insert_res(res2)

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


def test_get_loss_with_default_aggregator(
    populated_trainer_set_aggregate: TrainerSetAggregate,
):
    tsa = populated_trainer_set_aggregate

    aggregated_loss = tsa.get_loss()

    assert math.isinf(aggregated_loss)


def test_get_loss_with_custom_aggregator(
    populated_trainer_set_aggregate: TrainerSetAggregate,
):
    tsa = populated_trainer_set_aggregate

    aggregated_loss = tsa.get_loss(aggregator=min)

    assert aggregated_loss == pytest.approx(0.7967)


def test_get_loss_on_empty_returns_inf(
    empty_trainer_set_aggregate: TrainerSetAggregate,
):
    tsa = empty_trainer_set_aggregate
    assert math.isinf(tsa.get_loss())


def test_get_metric_with_custom_aggregator(
    populated_trainer_set_aggregate: TrainerSetAggregate,
):
    tsa = populated_trainer_set_aggregate

    def safe_mean(values):
        return statistics.mean(v for v in values if v is not None)

    metric_value = tsa.get_metric(name="r2", default=-1.0, aggregator=safe_mean)

    # The values for 'r2' are [0.83, None]. The safe_mean of this is 0.83.
    assert metric_value == pytest.approx(0.83)


def test_get_metric_non_existing_returns_default(
    populated_trainer_set_aggregate: TrainerSetAggregate,
):
    tsa = populated_trainer_set_aggregate

    def filter_none(values):
        return [v for v in values if v is not None]

    # The metric 'accuracy' does not exist in any result, so the aggregator
    # will receive [None, None] and return []. The `or default` kicks in.
    metric_value = tsa.get_metric(name="accuracy", default=-1.0, aggregator=filter_none)

    assert metric_value == -1.0


def test_get_metric_on_empty_returns_default(
    empty_trainer_set_aggregate: TrainerSetAggregate,
):
    tsa = empty_trainer_set_aggregate

    metric_value = tsa.get_metric(name="r2", default=-1.0)

    assert metric_value == -1.0
