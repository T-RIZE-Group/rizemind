from web3 import Web3
from eth_account import Account
from dotenv import load_dotenv
import os

def main():
    # Connect to local Anvil node
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    
    # Check connection
    if not w3.is_connected():
        print("Failed to connect to Ethereum node!")
        return
    
    # Load mnemonic from .env file
    load_dotenv()
    mnemonic = os.getenv("PRIVATE_SHAPLEY_MNEMONIC")
    if not mnemonic:
        print("No mnemonic found in .env file! Run generate_mnemonic.py first.")
        return
    
    # Enable HD wallet features
    Account.enable_unaudited_hdwallet_features()
    
    # Anvil's first account (which has lots of ETH by default)
    # Private key: 0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
    rich_account = w3.eth.account.from_key("0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80")
    
    # Get the aggregator and client addresses from mnemonic
    aggregator = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/0'/0/0")
    print(f"Funding aggregator: {aggregator.address}")
    
    # Transfer 1 ETH to the aggregator
    tx = {
        'from': rich_account.address,
        'to': aggregator.address,
        'value': w3.to_wei(1, 'ether'),
        'gas': 21000,
        'gasPrice': w3.eth.gas_price,
        'nonce': w3.eth.get_transaction_count(rich_account.address)
    }
    
    signed_tx = w3.eth.account.sign_transaction(tx, rich_account.key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print(f"Sent 1 ETH to aggregator. Transaction hash: {tx_hash.hex()}")
    
    # Fund 10 client accounts derived from the mnemonic
    for i in range(1, 11):
        client_path = f"m/44'/60'/{i}'/0/0"
        client_account = Account.from_mnemonic(mnemonic, account_path=client_path)
        print(f"Funding client {i}: {client_account.address}")
        
        tx = {
            'from': rich_account.address,
            'to': client_account.address,
            'value': w3.to_wei(0.5, 'ether'),
            'gas': 21000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(rich_account.address)
        }
        
        signed_tx = w3.eth.account.sign_transaction(tx, rich_account.key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Sent 0.5 ETH to client {i}. Transaction hash: {tx_hash.hex()}")

if __name__ == "__main__":
    main()