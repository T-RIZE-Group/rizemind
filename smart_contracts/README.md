# DML Smart Contracts

Refer to [ApeWorx.io](https://docs.apeworx.io/ape/latest/userguides/quickstart.html) for more documentation.

## Install

0. Create and active a venv
```shell
python -m venv sc_env
source sc_env/bin/activate
```

1. Install the repo dependencies
```shell

pip install -e .
```

2. Install Foundry
```shell
curl -L https://foundry.paradigm.xyz | bash
```

3. Install projects deps
```shell
ape plugins install
ape pm install
```

## upgrading plugins
Install project plugins. 
```shell
ape plugins install --upgrade .
```

## Compiling Contracts
Compiling uses [ape-solidity](https://github.com/ApeWorX/ape-solidity)
```shell
ape compile
```

## Writing Tests
Refer to [ApeWorx tests doc](https://docs.apeworx.io/ape/latest/userguides/testing.html)

## Running Local Devnet
Specify `--network ethereum:local:foundry`
1. Download Foundry
```shell
curl -L https://foundry.paradigm.xyz | bash
```
2. Install 
```shell
foundryup
```

## Make Smart Contracts available for DML Nodes

1. Start the foundry network
```
ape console --network ethereum:local:foundry
```
2. Deploy the contracts
```
ape run deploy MemberMgt --network ethereum:local:foundry --account TEST::0
```
3. Whitelist new addresses
```
ape run members add --network ethereum:local:foundry --account TEST::0 --member TEST::1 --contract <contract_address>
```
ape run deploy model_factory --network ethereum:local:foundry --account TEST::0
