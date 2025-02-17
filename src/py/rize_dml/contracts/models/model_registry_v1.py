from rize_dml.contracts.access_control.FlAccessControl import FlAccessControl
from rize_dml.contracts.deployed_contracts import load_contract_data
from rize_dml.contracts.models.model_registry import ModelRegistry
from web3 import Web3
from web3.contract import Contract


class ModelRegistryV1(FlAccessControl, ModelRegistry):
    def __init__(self, model: Contract):
        FlAccessControl.__init__(self, model)
        ModelRegistry.__init__(self, model)

    @staticmethod
    def from_address(address: str) -> "ModelRegistryV1":
        model = load_contract_data("ModelRegistryV1", "smart_contracts/output/local")
        checksum_address = Web3.to_checksum_address(address)
        w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
        return ModelRegistryV1(w3.eth.contract(address=checksum_address, abi=model.abi))
