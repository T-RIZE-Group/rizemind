import json
from pathlib import Path
from ape.contracts.base import ContractInstance


def save_contract_data(contract: ContractInstance, output_folder="output"):
    """Function to save contract data (address, chainId, ABI) to a JSON file."""
    # Prepare the data to write to the JSON file
    contract_data = {
        "address": contract.address,
        "abi": contract.contract_type.dict()["abi"],
        "name": contract.contract_type.name,
    }

    # Define the output file path
    output_path = Path(output_folder) / f"{contract.contract_type.name}.json"

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write data to the JSON file
    with open(output_path, "w") as f:
        json.dump(contract_data, f, indent=4)
