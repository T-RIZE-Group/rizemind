from rize_dml.contracts.access_control.FlAccessControl import FlAccessControl
from rize_dml.contracts.models.model_registry import ModelRegistry
from rize_dml.contracts.models.model_registry_v1 import ModelRegistryV1

def test_init():
  print(f"FlAccessControl metaclass: {type(FlAccessControl)}")
  print(f"ModelRegistry metaclass: {type(ModelRegistry)}")
  print(f"ModelRegistryV1 metaclass: {type(ModelRegistryV1)}")

  contract = ModelRegistryV1(None) 