=========
Generating and Storing a Mnemonic
=========

A mnemonic phrase, also known as a seed phrase, is a set of words used to recover a cryptocurrency wallet. It is crucial for securing access to blockchain-based accounts and should be handled with care.

Generating a Mnemonic
=====================

.. caution::
   Never share your mnemonic with anyone. Anyone with access to your mnemonic can gain full control over your assets. 
   Store it securely and avoid exposing it to untrusted software or online services.

Wallet-Based Option
-------------------

Most wallets, such as **MetaMask**, **Ledger Live**, and **Trezor Suite**, provide an option to generate a mnemonic phrase when setting up a new wallet. 

To generate a mnemonic using MetaMask:

1. Install the **MetaMask** browser extension or mobile app.

2. Follow the setup process and **securely write down** the generated 12 or 24-word seed phrase.

3. Store it in a safe place and never share it.

Using a Python Script
---------------------

You can generate a mnemonic programmatically using **rizemind** and **eth_account**:

.. code-block:: python

    from rizemind.authentication.config import AccountConfig
    from eth_account.hdaccount import generate_mnemonic

    def main():
        # Generate a new mnemonic phrase
        mnemonic = generate_mnemonic(lang="english", num_words=12)

        account = AccountConfig(mnemonic=mnemonic)

        # Print the mnemonic phrase
        print("Generated Mnemonic Phrase:")
        print(mnemonic)

        address = account.get_account(0).address
        print(f"Your aggregator address: {address}")

    if __name__ == "__main__":
        main()

When you execute this script, a new mnemonic will be randomly generated.

Storing Your Mnemonic
=====================

Since your mnemonic provides access to your account, it must be stored securely. Here are some recommended methods:

1. **Environment Variables**  
   Set an environment variable to store your mnemonic securely:

   .. code:: shell

       export MY_MNEMONIC="test test test test test test test test test test test junk"

2. **.env File (Unchecked from Git)**  
   Store your mnemonic in a `.env` file and **ensure it is excluded from version control (e.g., `.gitignore`)**:

   **.env file:**

   .. code:: text

       MY_MNEMONIC="test test test test test test test test test test test junk"

   Load it into your application using a library like `python-dotenv`:

   .. code-block:: python

       from dotenv import load_dotenv
       import os

       load_dotenv()
       mnemonic = os.getenv("MY_MNEMONIC")

       print(mnemonic)

3. **Hardware Wallets or Secure Password Managers**  
   Rizemind does not support those methods at the moment.

Using Mnemonic with rizemind
============================

The `TomlConfig` class from `rizemind` can parse a TOML config file and replace variables with environment variables.

Example usage:

1. **Set the mnemonic as an environment variable**:

   .. code:: shell

       export MY_MNEMONIC="test test test test test test test test test test test junk"

2. **Define a TOML configuration file (`myconfig.toml`)**:

   .. code:: toml

       [my-account]
       mnemonic="$MY_MNEMONIC"

3. **Load and retrieve the mnemonic in Python**:

   .. code-block:: python

       from dotenv import load_dotenv
       from rizemind.configuration.toml_config import TomlConfig

       load_dotenv()
       config = TomlConfig('./myconfig.toml')
       mnemonic = config.get("my-account.mnemonic")
       print(mnemonic)  # result: test test test test test test test test test test test junk
