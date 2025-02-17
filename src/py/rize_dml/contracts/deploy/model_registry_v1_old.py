from typing import List
from eth_account import Account
from rize_dml.contracts import load_compile
from web3 import Web3
from web3.contract import Contract


class TrainerGroup:
    contract: Contract
    chain_id: int
    _name: str = None

    def __init__(self, contract: Contract, chain_id: int):
        self.contract = contract
        self.chain_id = chain_id

    def validate_signer(self, address: str) -> bool:
        return self.contract.functions.isWhitelisted(address).call()

    def name(self) -> str:
        if self._name is None:
            self._name = self.contract.functions.name().call()
        return self._name


def deploy_group(
    deployer: Account, name: str, member_address: List[str]
) -> TrainerGroup:
    # 1. Compile the Solidity Contract
    contract = load_compile("MemberManagement.sol", "MemberManagement")

    # 2. Connect to an Ethereum Node
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    assert w3.is_connected(), "Failed to connect to the Ethereum node."

    # 3. Deploy the Contract with Constructor Arguments
    contract = w3.eth.contract(abi=contract.abi, bytecode=contract.bytecode)
    # Build the transaction
    tx = contract.constructor(name, member_address, 0).build_transaction(
        {
            "from": deployer.address,
            "nonce": w3.eth.get_transaction_count(deployer.address),
            "gas": 2000000,
            "gasPrice": w3.to_wei("20", "gwei"),
        }
    )

    # Sign the transaction
    signed_tx = deployer.sign_transaction(tx)

    # Send the transaction
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

    # Wait for the transaction to be mined
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    assert tx_receipt.status != 0, "Deployment transaction failed or reverted."
    contract_address = tx_receipt.contractAddress
    print("contract address", contract_address)
    contract = w3.eth.contract(address=contract_address, abi=contract.abi)
    return TrainerGroup(contract, w3.eth.chain_id)
