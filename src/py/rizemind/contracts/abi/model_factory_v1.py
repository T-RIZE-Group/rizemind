import json

abi_json = """[
        {
            "type": "constructor",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "logicContract",
                    "type": "address",
                    "internalType": "address"
                }
            ]
        },
        {
            "type": "error",
            "name": "AccessControlBadConfirmation",
            "inputs": []
        },
        {
            "type": "error",
            "name": "AccessControlUnauthorizedAccount",
            "inputs": [
                {
                    "name": "account",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "neededRole",
                    "type": "bytes32",
                    "internalType": "bytes32"
                }
            ]
        },
        {
            "type": "event",
            "name": "ContractCreated",
            "inputs": [
                {
                    "name": "proxyAddress",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                },
                {
                    "name": "owner",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                },
                {
                    "name": "name",
                    "type": "string",
                    "internalType": "string",
                    "indexed": false
                }
            ],
            "anonymous": false
        },
        {
            "type": "event",
            "name": "ProxyUpgraded",
            "inputs": [
                {
                    "name": "proxyAddress",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                },
                {
                    "name": "newLogic",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                }
            ],
            "anonymous": false
        },
        {
            "type": "event",
            "name": "RoleAdminChanged",
            "inputs": [
                {
                    "name": "role",
                    "type": "bytes32",
                    "internalType": "bytes32",
                    "indexed": true
                },
                {
                    "name": "previousAdminRole",
                    "type": "bytes32",
                    "internalType": "bytes32",
                    "indexed": true
                },
                {
                    "name": "newAdminRole",
                    "type": "bytes32",
                    "internalType": "bytes32",
                    "indexed": true
                }
            ],
            "anonymous": false
        },
        {
            "type": "event",
            "name": "RoleGranted",
            "inputs": [
                {
                    "name": "role",
                    "type": "bytes32",
                    "internalType": "bytes32",
                    "indexed": true
                },
                {
                    "name": "account",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                },
                {
                    "name": "sender",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                }
            ],
            "anonymous": false
        },
        {
            "type": "event",
            "name": "RoleRevoked",
            "inputs": [
                {
                    "name": "role",
                    "type": "bytes32",
                    "internalType": "bytes32",
                    "indexed": true
                },
                {
                    "name": "account",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                },
                {
                    "name": "sender",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                }
            ],
            "anonymous": false
        },
        {
            "type": "function",
            "name": "DEFAULT_ADMIN_ROLE",
            "stateMutability": "view",
            "inputs": [],
            "outputs": [
                {
                    "name": "",
                    "type": "bytes32",
                    "internalType": "bytes32"
                }
            ]
        },
        {
            "type": "function",
            "name": "createModel",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "name",
                    "type": "string",
                    "internalType": "string"
                },
                {
                    "name": "symbol",
                    "type": "string",
                    "internalType": "string"
                },
                {
                    "name": "aggregator",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "initialTrainers",
                    "type": "address[]",
                    "internalType": "address[]"
                }
            ],
            "outputs": []
        },
        {
            "type": "function",
            "name": "getRoleAdmin",
            "stateMutability": "view",
            "inputs": [
                {
                    "name": "role",
                    "type": "bytes32",
                    "internalType": "bytes32"
                }
            ],
            "outputs": [
                {
                    "name": "",
                    "type": "bytes32",
                    "internalType": "bytes32"
                }
            ]
        },
        {
            "type": "function",
            "name": "grantRole",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "role",
                    "type": "bytes32",
                    "internalType": "bytes32"
                },
                {
                    "name": "account",
                    "type": "address",
                    "internalType": "address"
                }
            ],
            "outputs": []
        },
        {
            "type": "function",
            "name": "hasRole",
            "stateMutability": "view",
            "inputs": [
                {
                    "name": "role",
                    "type": "bytes32",
                    "internalType": "bytes32"
                },
                {
                    "name": "account",
                    "type": "address",
                    "internalType": "address"
                }
            ],
            "outputs": [
                {
                    "name": "",
                    "type": "bool",
                    "internalType": "bool"
                }
            ]
        },
        {
            "type": "function",
            "name": "implementation",
            "stateMutability": "nonpayable",
            "inputs": [],
            "outputs": [
                {
                    "name": "",
                    "type": "address",
                    "internalType": "address"
                }
            ]
        },
        {
            "type": "function",
            "name": "renounceRole",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "role",
                    "type": "bytes32",
                    "internalType": "bytes32"
                },
                {
                    "name": "callerConfirmation",
                    "type": "address",
                    "internalType": "address"
                }
            ],
            "outputs": []
        },
        {
            "type": "function",
            "name": "revokeRole",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "role",
                    "type": "bytes32",
                    "internalType": "bytes32"
                },
                {
                    "name": "account",
                    "type": "address",
                    "internalType": "address"
                }
            ],
            "outputs": []
        },
        {
            "type": "function",
            "name": "supportsInterface",
            "stateMutability": "view",
            "inputs": [
                {
                    "name": "interfaceId",
                    "type": "bytes4",
                    "internalType": "bytes4"
                }
            ],
            "outputs": [
                {
                    "name": "",
                    "type": "bool",
                    "internalType": "bool"
                }
            ]
        },
        {
            "type": "function",
            "name": "updateImplementation",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "implementation",
                    "type": "address",
                    "internalType": "address"
                }
            ],
            "outputs": []
        }
    ]
"""

model_factory_v1_abi = json.loads(abi_json)
