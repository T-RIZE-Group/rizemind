==============================================
How To Add Web3-Based Signature Authentication
==============================================

This guide will walk you through implementing **Web3-based signature authentication** using `rizemind` with **Flower**, ensuring secure, blockchain-validated authentication for federated learning.

**Requirements**

- Follow the :doc:`Generating and Storing a Mnemonic <../generate-store-mnemonic>` tutorial, as you will need a mnemonic.

- Have a working **Flower** project.

We recommend installing `python-dotenv` to store your mnemonic in a `.env` file:

.. code:: shell

   # If you use pip
   pip install python-dotenv

   # If you use uv
   uv add python-dotenv

Modifying the Client
====================

We will use the `SigningClient` class, which seamlessly integrates with any existing **FlowerClient**. This class pulls signing instructions from the blockchain. To achieve this, we must define:
1. The **gateway URL** to access the blockchain.
2. The **mnemonic** to sign updates.

### Adding Configuration

Add these fields to your `pyproject.toml`:

.. code-block:: toml

   [tool.eth.account]
   mnemonic = "$RIZENET_MNEMONIC"

   [tool.web3]
   url = "https://testnet.rizenet.io"

Next, create a `.env` file in the same directory as `pyproject.toml` to store your secrets securely (**never commit this file to Git**):

.. code-block:: text

   RIZENET_MNEMONIC="put your mnemonic here"

.. note::
   `TomlConfig` will automatically replace values in the config with environment variables.

### Updating the Client

Modify the client function to use `SigningClient`:

.. code-block:: python

   from dotenv import load_dotenv
   from rizemind.authentication.config import AccountConfig
   from rizemind.configuration.toml_config import TomlConfig
   from rizemind.web3.config import Web3Config
   from rizemind.authentication.eth_account_client import SigningClient
   from .task import load_data, load_model

   def client_fn(context: Context):
       """Construct a Client that will be run in a ClientApp."""

       # Read node_config to fetch partition details
       partition_id = int(context.node_config["partition-id"])
       num_partitions = int(context.node_config["num-partitions"])
       data = load_data(partition_id, num_partitions)

       # Read run_config to fetch hyperparameters
       epochs = context.run_config["local-epochs"]
       batch_size = context.run_config["batch-size"]
       verbose = context.run_config.get("verbose")
       learning_rate = context.run_config["learning-rate"]

       client = FlowerClient(learning_rate, data, epochs, batch_size, verbose).to_client()

       ##########
       # Adding signing support
       ##########

       # Load values from .env into os.environ
       load_dotenv()

       # Load config and parse env variables
       config = TomlConfig("./pyproject.toml")

       # Create an AccountConfig from the TOML file
       account_config = AccountConfig(**config.get("tool.eth.account"))

       # Derive an address from the mnemonic using HD path
       account = account_config.get_account(partition_id + 1)

       # Load blockchain gateway configuration
       web3_config = Web3Config(**config.get("tool.web3"))

       # Return Client instance with signing capability
       return SigningClient(
           client,
           account,
           web3_config.get_web3(),
       )

Modifying the Aggregator
========================

On the **aggregator** side, we will use `EthAccountStrategy` to validate Web3-based signatures. This integrates seamlessly with any **FlowerStrategy**.

Adding Smart Contract Configuration
-----------------------------------

Add the following parameters to configure the **model's smart contract** in `pyproject.toml`:

.. code-block:: toml

   [tool.web3.model_v1]
   name = "test_model"
   ticker = "tst"

Updating the Server
-------------------

Modify `server_fn` to integrate signature validation:

.. code-block:: python

   def server_fn(context: Context):
       """Construct components that set the ServerApp behaviour."""
       parameters = ndarrays_to_parameters(load_model().get_weights())

       # Define the strategy
       strategy = FedAvg(
           fraction_fit=float(context.run_config["fraction-fit"]),
           fraction_evaluate=1.0,
           min_available_clients=2,
           initial_parameters=parameters,
           evaluate_metrics_aggregation_fn=weighted_average,
       )

       # Read config values
       num_rounds = int(context.run_config["num-server-rounds"])

       #######
       # Adding signature authentication
       #######

       # Load .env values into os.environ
       load_dotenv()

       # Load and parse config
       config = TomlConfig("./pyproject.toml")

       # Load account and blockchain configuration
       auth_config = AccountConfig(**config.get("tool.eth.account"))
       web3_config = Web3Config(**config.get("tool.web3"))
       w3 = web3_config.get_web3()

       # Derive the aggregator account (account 0)
       account = auth_config.get_account(0)

       # Generate trainer accounts
       members = [auth_config.get_account(i).address for i in range(1, 11)]

       # Load the model configuration
       model_v1_config = ModelFactoryV1Config(**config.get("tool.web3.model_v1"))

       # Deploy the smart contract
       contract = ModelFactoryV1(model_v1_config).deploy(account, members, w3)

       # Define server configuration
       config = ServerConfig(num_rounds=int(num_rounds))

       # Enable authentication strategy
       auth_strategy = EthAccountStrategy(strategy, contract)

       return ServerAppComponents(strategy=auth_strategy, config=config)

Run to Test
===========

Run your Flower project with:

.. code:: shell

   flwr run .

Debugging
---------

**Issue: Account cannot deploy contracts**
   - Copy the **address in the error message**.
   - Visit `rizenet.io/deployer <https://rizenet.io/deployer>`_ and follow the steps to **add the address to the whitelist**.

**Issue: Account does not have enough gas**
   - Visit `rizenet.io/faucets <https://rizenet.io/faucets>`_ to get free **testnet gas**.

----

By following these steps, you have successfully added **Web3-based signature authentication** to your Flower project, ensuring secure client authentication and model integrity on the blockchain.
