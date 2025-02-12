from pydantic import BaseModel
from eth_pydantic_types import Address
from typing import List
from pathlib import Path
import json

class DeployedContract(BaseModel):
    address: Address
    abi: List[dict]

  
def load_contract_data(contract_name, output_folder='output') -> DeployedContract:
    """Function to load contract data from a JSON file into a ContractData Pydantic model."""

      # Get the directory of the current Python file
    current_dir = Path(__file__).parent

    # Construct the path to the Solidity file
    output_folder = current_dir / f'../../../../smart_contracts/output/local'
    output_folder = output_folder.resolve()  # Resolve to an absolute path

    input_path = Path(output_folder) / f"{contract_name}.json"
  

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    with open(input_path, 'r') as f:
        file_data = json.load(f)
    
    return DeployedContract.model_validate(file_data)