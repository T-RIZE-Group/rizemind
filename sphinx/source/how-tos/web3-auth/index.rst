=====
How To Add Web3 Based Signature Authentication
=====

**Requirements**
- Follow the Generating and Storing Mnemonic tutorial as you will
need a mnemonic.
- Have a working Flower Project.

We recommend you install python-dotenv to store a menmonic in a .env file.

```pip install python-dotenv```

Modifying the Client
====================

We will use the SigningClient class which seamlessly integrates with any
existing FlowerClient. This class will pull the signing instructions from
the blockchain. For this reason, we need to define the gateway to access the
blockchain and the mnemonic to sign the updates.

Add these fields to your `pyproject.toml`.

.. code-block:: toml
  [tool.eth.account]
  mnemonic = "$RIZENET_MNEMONIC"

  [tool.web3]
  url = "https://testnet.rizenet.io"

in a `.env` in the same folder than the `pyproject.toml`. You should store 
secrets in this file, but do not commit it to git or you risk loosing any funds/model tokens,
permissions to the models deployed with the mnemonic.

.. code-block:: .env
  RIZENET_MNEMONIC="put your mnemonic here"

[reminder: the TomlConfig will parse values in the config to replace them with environment variable]


Now we need to modify the 

.. code-block:: python
  from dotenv import load_dotenv

  from rizemind.authentication.config import AccountConfig
  from rizemind.configuration.toml_config import TomlConfig
  from rizemind.web3.config import Web3Config
  from .task import load_data, load_model
  from rizemind.authentication.eth_account_client import SigningClient

  def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    data = load_data(partition_id, num_partitions)

    # Read run_config to fetch hyperparameters relevant to this run
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")
    learning_rate = context.run_config["learning-rate"]
    client = FlowerClient(learning_rate, data, epochs, batch_size, verbose).to_client()

    ##########
    # following is the modifications to support signing
    ##########
    # load values from the .env file in os.environ
    load_dotenv()
    # Load the configuration and parse env variables
    config = TomlConfig("./pyproject.toml")
    # Creates a AccountConfig using the config section
    account_config = AccountConfig(**config.get("tool.eth.account"))
    # Derives and address of the mnemonic using HD path
    account = account_config.get_account(partition_id + 1)
    # Loads the gateway information
    web3_config = Web3Config(**config.get("tool.web3"))

    # Return Client instance
    return SigningClient(
        client,
        account,
        web3_config.get_web3(),
    )

Modifying the Strategy
=======================

On the aggregator side, we will use the EthAccountStrategy which
can integrate with any FlowerStrategy.

Start by adding the following parameters to configure the model's
smart contract.

.. code-block:: toml
  [tool.web3.model_v1]
  name = "test_model"
  ticker = "tst"


Now we modify the `server_fn` to integrate the signature validation.

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
    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])
    #######
    # Modifications to filter whitelisted trainers
    #######
    # load .env variables into os.environ
    load_dotenv()
    # load config and parses env variables
    config = TomlConfig("./pyproject.toml")
    # loads the account config
    auth_config = AccountConfig(**config.get("tool.eth.account"))
    # loads the gateway config
    web3_config = Web3Config(**config.get("tool.web3"))
    # gets web3 instance
    w3 = web3_config.get_web3()
    # derives the account 0 which will be the aggregator
    account = auth_config.get_account(0)
    members = []
    # derives the trainers account addresses. 
    # In production, you would already have the addresses or add them post-deployment
    for i in range(1, 11):
        trainer = auth_config.get_account(i)
        members.append(trainer.address)

    # loads the config for the model
    model_v1_config = ModelFactoryV1Config(**config.get("tool.web3.model_v1"))
    # deploys the smart contract
    contract = ModelFactoryV1(model_v1_config).deploy(account, members, w3)
    config = ServerConfig(num_rounds=int(num_rounds))
    authStrategy = EthAccountStrategy(
        strategy, contract
    )
    return ServerAppComponents(strategy=authStrategy, config=config)


Run to test
===========

Run your flower project with ``flwr run .``

Debugging
---------

**Account cannot deploy contracts**

In this case, copy the address in the error message and head to rizenet.io/deployer 
and follow the steps to add the address to the whitelist.

**Account does not have enough gas**

You can get testnet gas for free at rizenet.io/faucets