from eth_account.hdaccount import generate_mnemonic
from rizemind.authentication.config import AccountConfig


def main():
    # Generate a new mnemonic phrase
    mnemonic = generate_mnemonic(lang="english", num_words=12)

    account = AccountConfig(mnemonic=mnemonic)

    # Print the mnemonic phrase
    print("Generated Mnemonic Phrase:")
    print(mnemonic)
    print(
        "Copy the `env.example` to `.env` and replace the RIZENET_MNEMONIC with the mnemonic above"
    )

    address = account.get_account(0).address
    print(f"Your aggregator address: {address}")
    print("To enable your aggregator to launch contracts on Rizenet:")
    print("1. Enable smart contract deployment: https://rizenet.io/deployer")
    print("2. Get gas: https://rizenet.io/faucets")


if __name__ == "__main__":
    main()
