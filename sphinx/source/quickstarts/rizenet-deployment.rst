==========
Deploying on Rizenet Testnet
==========

This quickstart will help you set up and deploy on the **Rizenet Testnet**.

Rizenet provides **Factories**, which enable an aggregator to easily instantiate new **smart contracts** for **access control** and **compensation distribution** in decentralized machine learning (DML) systems.

Clone the Repository
====================

First, clone the `dml` repository and navigate to the deployment example:

.. code-block:: shell

    git clone git@github.com:T-RIZE-Group/dml.git
    cd dml/examples/rizenet_deployment

Generate a Mnemonic
===================

A **mnemonic phrase** (also known as a seed phrase) is required to manage blockchain accounts.

Run the following command to generate one:

.. code-block:: shell

    python rizenet_deployment/generate_mnemonic.py

The output will look something like this:

.. code-block:: text

    Generated Mnemonic Phrase:
    picture fine relief success curious avocado define divert cause genuine such master

    Copy the `env.example` to `.env` and replace the RIZENET_MNEMONIC with the mnemonic above.

    Your aggregator address: 0x16E5D170372bDEF845dE2192429e822900F7592F

    To enable your aggregator to launch contracts on Rizenet:
    1. Enable smart contract deployment: https://rizenet.io/deployer
    2. Get gas: https://rizenet.io/faucets

The two most important pieces of information here are:
- **The 12-word mnemonic phrase** (used to derive private keys).
- **The aggregator address** (used to deploy contracts).

Save Your Mnemonic
==================

To securely store your mnemonic, copy `env.example` to `.env`:

.. code-block:: shell

    cp env.example .env

Then, **edit the `.env` file** and replace the placeholder with your generated mnemonic:

.. code-block:: text

    MNEMONIC="picture fine relief success curious avocado define divert cause genuine such master"

.. warning::
   **Never share your mnemonic with anyone.** It grants full access to your accounts. Store it securely and do not commit `.env` files to version control.

Whitelist Aggregator
====================

Rizenet is a **permissioned network** for **smart contract deployment**.  
Before deploying contracts, you must **whitelist your aggregator address**.

Steps to Whitelist:
-------------------
1. Go to `Rizenet Deployer <https://rizenet.io/deployer>`_.
2. Enter your **aggregator address** (generated earlier).
3. Click "Enable" and wait for confirmation.

Once approved, your aggregator will have permission to deploy smart contracts.

Get Gas
====================

Blockchain transactions require **gas** to process computations.  
On the **Rizenet Testnet**, gas is **free**, but you need to request testnet tokens.

Steps to Get Gas:
-----------------
1. Visit `Rizenet Faucet <https://rizenet.io/faucets>`_.
2. Enter your **aggregator address**.
3. Click drip and wait for confirmation.

Once your aggregator has gas, you're ready to deploy smart contracts.

Run the project
================

Run the project.

.. code-block:: shell

    flwr run .

For more details on using Web3-based authentication, see the :doc:`How To Add Web3-Based Signature Authentication <../how-tos/web3-auth/index>`.
