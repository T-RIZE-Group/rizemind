"""Generate a mnemonic for private Shapley example."""

from eth_account.hdaccount import generate_mnemonic
from eth_account import Account
import os
from pathlib import Path

def main():
    """Generate a mnemonic and save it to .env file."""
    # Enable HD wallet features
    Account.enable_unaudited_hdwallet_features()
    
    # Generate a new mnemonic phrase
    mnemonic = generate_mnemonic(lang="english", num_words=12)
    
    # Print the mnemonic phrase
    print("Generated Mnemonic Phrase:")
    print(mnemonic)
    
    # Get the aggregator address
    hd_path = "m/44'/60'/0'/0/0"
    account = Account.from_mnemonic(mnemonic, account_path=hd_path)
    aggregator_address = account.address
    
    print(f"Aggregator address: {aggregator_address}")
    
    # Get the first client address
    client_hd_path = "m/44'/60'/1'/0/0"
    client_account = Account.from_mnemonic(mnemonic, account_path=client_hd_path)
    client_address = client_account.address
    
    print(f"First client address: {client_address}")
    
    # Create or update .env file
    env_path = Path("../.env")
    
    # Check if .env exists and contains PRIVATE_SHAPLEY_MNEMONIC
    if env_path.exists():
        with open(env_path, "r") as f:
            content = f.read()
            if "PRIVATE_SHAPLEY_MNEMONIC" in content:
                print(".env file already contains PRIVATE_SHAPLEY_MNEMONIC, not modifying")
                return
    
    # Write to .env file
    with open(env_path, "a+") as f:
        f.write(f"\nPRIVATE_SHAPLEY_MNEMONIC=\"{mnemonic}\"\n")
    
    print(f"Mnemonic saved to {env_path.absolute()}")
    print("\nTo use this with Anvil, you'll need to fund these addresses:")
    print(f"- Aggregator: {aggregator_address}")
    print(f"- First client: {client_address}")

if __name__ == "__main__":
    main()