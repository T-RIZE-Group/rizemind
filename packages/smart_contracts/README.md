# DML Smart Contracts

Refer to [ApeWorx.io](https://docs.apeworx.io/ape/latest/userguides/quickstart.html) for more documentation.

## Install

1. Install the framework CLI
```shell

pipx install eth-ape
```

2. Install Foundry
```shell
curl -L https://foundry.paradigm.xyz | bash
```

3. Install project plugins. (If `ape` is not found, just close and reopen the shell)
```shell
ape plugins install .
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
