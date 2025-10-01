from flwr.common import Config
from hexbytes import HexBytes
from pydantic import BaseModel
from rizemind.configuration.transform import from_config, to_config
from rizemind.exception.parse_exception import catch_parse_errors

EVALUATION_TASK_INS_PREFIX = "rizemind.evaluation.task.ins"


class EvaluationTaskIns(BaseModel):
    round_id: int
    eval_id: int
    set_id: int
    model_hash: HexBytes


def prepare_evaluation_task_ins(
    *,
    round_id: int,
    eval_id: int,
    set_id: int,
    model_hash: HexBytes,
) -> Config:
    payload = EvaluationTaskIns(
        round_id=round_id,
        eval_id=eval_id,
        set_id=set_id,
        model_hash=model_hash,
    )
    return to_config(payload.model_dump(), prefix=EVALUATION_TASK_INS_PREFIX)


@catch_parse_errors
def parse_evaluation_task_ins(config: Config) -> EvaluationTaskIns:
    payload = from_config(config)
    return EvaluationTaskIns(**payload["rizemind"]["evaluation"]["task"]["ins"])
