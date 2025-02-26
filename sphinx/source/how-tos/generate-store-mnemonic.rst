=========
Generating and Storing a Mnemonic
=========

[add a mnemonic introduction]

Generating a Mnemonic
=====================

[caution with leaking mnemonic]

[Wallet based option with Metamask, ect]

Using a Python script:

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


When you execute it, a new mnemonic will be randomly generated.

Storing Your Mnemonic
=====================

[You can store the mnemonic in a .env file unchecked from your git repo]
[store it in env variable]

Using Mnemonic with rizemind
============================

The TomlConfig class will parse your config file to replace variables
environment variables.

.. code:: shell
    EXPORT MY_MNEMONIC="test test test test test test test test test test test junk"

.. code:: toml
    [my-account]
    mnemonic="$MY_MNEMONIC"

.. code:: python
    from rizemind.configuration.toml_config import TomlConfig

    config = TomlConfig('./myconfig.toml')
    mnemonic = config.get("my-account.mnemonic")
    print(mnemonic) # result: test test test test test test test test test test test junk