from .deployed_contracts import load_contract_data

def test_load_contract_data():
  load_contract_data("ModelRegistryFactory","smart_contracts/output/local")
