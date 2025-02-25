import json

abi_json = """[
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
            "type": "error",
            "name": "BadRewards",
            "inputs": []
        },
        {
            "type": "error",
            "name": "ERC20InsufficientAllowance",
            "inputs": [
                {
                    "name": "spender",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "allowance",
                    "type": "uint256",
                    "internalType": "uint256"
                },
                {
                    "name": "needed",
                    "type": "uint256",
                    "internalType": "uint256"
                }
            ]
        },
        {
            "type": "error",
            "name": "ERC20InsufficientBalance",
            "inputs": [
                {
                    "name": "sender",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "balance",
                    "type": "uint256",
                    "internalType": "uint256"
                },
                {
                    "name": "needed",
                    "type": "uint256",
                    "internalType": "uint256"
                }
            ]
        },
        {
            "type": "error",
            "name": "ERC20InvalidApprover",
            "inputs": [
                {
                    "name": "approver",
                    "type": "address",
                    "internalType": "address"
                }
            ]
        },
        {
            "type": "error",
            "name": "ERC20InvalidReceiver",
            "inputs": [
                {
                    "name": "receiver",
                    "type": "address",
                    "internalType": "address"
                }
            ]
        },
        {
            "type": "error",
            "name": "ERC20InvalidSender",
            "inputs": [
                {
                    "name": "sender",
                    "type": "address",
                    "internalType": "address"
                }
            ]
        },
        {
            "type": "error",
            "name": "ERC20InvalidSpender",
            "inputs": [
                {
                    "name": "spender",
                    "type": "address",
                    "internalType": "address"
                }
            ]
        },
        {
            "type": "error",
            "name": "InvalidInitialization",
            "inputs": []
        },
        {
            "type": "error",
            "name": "NotInitializing",
            "inputs": []
        },
        {
            "type": "error",
            "name": "RoundMismatch",
            "inputs": [
                {
                    "name": "currentRound",
                    "type": "uint256",
                    "internalType": "uint256"
                },
                {
                    "name": "givenRound",
                    "type": "uint256",
                    "internalType": "uint256"
                }
            ]
        },
        {
            "type": "event",
            "name": "Approval",
            "inputs": [
                {
                    "name": "owner",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                },
                {
                    "name": "spender",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                },
                {
                    "name": "value",
                    "type": "uint256",
                    "internalType": "uint256",
                    "indexed": false
                }
            ],
            "anonymous": false
        },
        {
            "type": "event",
            "name": "EIP712DomainChanged",
            "inputs": [],
            "anonymous": false
        },
        {
            "type": "event",
            "name": "Initialized",
            "inputs": [
                {
                    "name": "version",
                    "type": "uint64",
                    "internalType": "uint64",
                    "indexed": false
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
            "type": "event",
            "name": "RoundFinished",
            "inputs": [
                {
                    "name": "roundId",
                    "type": "uint256",
                    "internalType": "uint256",
                    "indexed": true
                },
                {
                    "name": "trainer",
                    "type": "uint64",
                    "internalType": "uint64",
                    "indexed": false
                },
                {
                    "name": "modelScore",
                    "type": "uint64",
                    "internalType": "uint64",
                    "indexed": false
                },
                {
                    "name": "totalContribution",
                    "type": "uint128",
                    "internalType": "uint128",
                    "indexed": false
                }
            ],
            "anonymous": false
        },
        {
            "type": "event",
            "name": "Transfer",
            "inputs": [
                {
                    "name": "from",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                },
                {
                    "name": "to",
                    "type": "address",
                    "internalType": "address",
                    "indexed": true
                },
                {
                    "name": "value",
                    "type": "uint256",
                    "internalType": "uint256",
                    "indexed": false
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
            "name": "allowance",
            "stateMutability": "view",
            "inputs": [
                {
                    "name": "owner",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "spender",
                    "type": "address",
                    "internalType": "address"
                }
            ],
            "outputs": [
                {
                    "name": "",
                    "type": "uint256",
                    "internalType": "uint256"
                }
            ]
        },
        {
            "type": "function",
            "name": "approve",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "spender",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "value",
                    "type": "uint256",
                    "internalType": "uint256"
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
            "name": "balanceOf",
            "stateMutability": "view",
            "inputs": [
                {
                    "name": "account",
                    "type": "address",
                    "internalType": "address"
                }
            ],
            "outputs": [
                {
                    "name": "",
                    "type": "uint256",
                    "internalType": "uint256"
                }
            ]
        },
        {
            "type": "function",
            "name": "canTrain",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "trainer",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "roundId",
                    "type": "uint256",
                    "internalType": "uint256"
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
            "name": "curentRound",
            "stateMutability": "view",
            "inputs": [],
            "outputs": [
                {
                    "name": "",
                    "type": "uint256",
                    "internalType": "uint256"
                }
            ]
        },
        {
            "type": "function",
            "name": "decimals",
            "stateMutability": "view",
            "inputs": [],
            "outputs": [
                {
                    "name": "",
                    "type": "uint8",
                    "internalType": "uint8"
                }
            ]
        },
        {
            "type": "function",
            "name": "distribute",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "trainers",
                    "type": "address[]",
                    "internalType": "address[]"
                },
                {
                    "name": "contributions",
                    "type": "uint64[]",
                    "internalType": "uint64[]"
                }
            ],
            "outputs": []
        },
        {
            "type": "function",
            "name": "eip712Domain",
            "stateMutability": "view",
            "inputs": [],
            "outputs": [
                {
                    "name": "fields",
                    "type": "bytes1",
                    "internalType": "bytes1"
                },
                {
                    "name": "name",
                    "type": "string",
                    "internalType": "string"
                },
                {
                    "name": "version",
                    "type": "string",
                    "internalType": "string"
                },
                {
                    "name": "chainId",
                    "type": "uint256",
                    "internalType": "uint256"
                },
                {
                    "name": "verifyingContract",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "salt",
                    "type": "bytes32",
                    "internalType": "bytes32"
                },
                {
                    "name": "extensions",
                    "type": "uint256[]",
                    "internalType": "uint256[]"
                }
            ]
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
            "name": "initialize",
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
            "name": "isAggregator",
            "stateMutability": "view",
            "inputs": [
                {
                    "name": "aggregator",
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
            "name": "isTrainer",
            "stateMutability": "view",
            "inputs": [
                {
                    "name": "trainer",
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
            "name": "name",
            "stateMutability": "view",
            "inputs": [],
            "outputs": [
                {
                    "name": "",
                    "type": "string",
                    "internalType": "string"
                }
            ]
        },
        {
            "type": "function",
            "name": "nextRound",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "summary",
                    "type": "tuple",
                    "components": [
                        {
                            "name": "roundId",
                            "type": "uint256",
                            "internalType": "uint256"
                        },
                        {
                            "name": "nTrainers",
                            "type": "uint64",
                            "internalType": "uint64"
                        },
                        {
                            "name": "modelScore",
                            "type": "uint64",
                            "internalType": "uint64"
                        },
                        {
                            "name": "totalContributions",
                            "type": "uint128",
                            "internalType": "uint128"
                        }
                    ],
                    "internalType": "struct RoundSummary"
                }
            ],
            "outputs": []
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
            "name": "symbol",
            "stateMutability": "view",
            "inputs": [],
            "outputs": [
                {
                    "name": "",
                    "type": "string",
                    "internalType": "string"
                }
            ]
        },
        {
            "type": "function",
            "name": "totalSupply",
            "stateMutability": "view",
            "inputs": [],
            "outputs": [
                {
                    "name": "",
                    "type": "uint256",
                    "internalType": "uint256"
                }
            ]
        },
        {
            "type": "function",
            "name": "transfer",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "to",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "value",
                    "type": "uint256",
                    "internalType": "uint256"
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
            "name": "transferFrom",
            "stateMutability": "nonpayable",
            "inputs": [
                {
                    "name": "from",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "to",
                    "type": "address",
                    "internalType": "address"
                },
                {
                    "name": "value",
                    "type": "uint256",
                    "internalType": "uint256"
                }
            ],
            "outputs": [
                {
                    "name": "",
                    "type": "bool",
                    "internalType": "bool"
                }
            ]
        }
    ]
"""

model_abi_v1_0_0 = json.loads(abi_json)
