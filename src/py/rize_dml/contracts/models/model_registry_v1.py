from rize_dml.contracts.access_control.FlAccessControl import FlAccessControl
from rize_dml.contracts.models.model_registry import ModelRegistry
from web3.contract import Contract


class ModelRegistryV1(FlAccessControl, ModelRegistry):
  def __init__(self, model: Contract):
    super().__init__(model)