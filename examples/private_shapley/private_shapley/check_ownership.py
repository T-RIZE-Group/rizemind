# check_ownership.py
from web3 import Web3
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()
    contract_address = os.getenv("PRIVATE_SHAPLEY_CONTRACT")
    
    if not contract_address:
        print("ERROR: PRIVATE_SHAPLEY_CONTRACT environment variable not set!")
        return
    
    # Connect to local Anvil node
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    
    # Check connection
    if not w3.is_connected():
        print("Failed to connect to Ethereum node!")
        return
    
    print(f"Connected to Ethereum node. Chain ID: {w3.eth.chain_id}")
    
    # Check contract owner
    # We need the owner() function ABI
    abi = [
        {
            "inputs": [],
            "name": "owner",
            "outputs": [{"name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]
    
    contract = w3.eth.contract(address=contract_address, abi=abi)
    
    try:
        owner = contract.functions.owner().call()
        print(f"Contract owner: {owner}")
        
        # Print the aggregator address from the .env file to compare
        mnemonic = os.getenv("PRIVATE_SHAPLEY_MNEMONIC")
        
        if mnemonic:
            from eth_account import Account
            Account.enable_unaudited_hdwallet_features()
            aggregator = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/0'/0/0")
            print(f"Your aggregator address: {aggregator.address}")
            
            if owner.lower() != aggregator.address.lower():
                print("WARNING: Contract owner is NOT your aggregator address!")
                print("This is why registerTrainer() calls are failing.")
            else:
                print("Your aggregator IS the contract owner, should have permission.")
    except Exception as e:
        print(f"Error checking ownership: {e}")

if __name__ == "__main__":
    main()