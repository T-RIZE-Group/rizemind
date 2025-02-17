import flwr as fl
from web3 import Web3

# import ipfshttpclient
import json
import pickle
import ipfshttpclient
import csv
import time

# Smart contract address and ABI
contract_address = "0x0165878A594ca255338adfa4d48449f69242Eb8F"
web3 = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:7545"))
MemberMgtAbi = json.loads(""" [
    
                
      {
          "inputs": [
            {
              "internalType": "address[]",
              "name": "initialMembers",
              "type": "address[]"
            },
            {
              "internalType": "uint256",
              "name": "initialThreshold",
              "type": "uint256"
            }
          ],
          "stateMutability": "nonpayable",
          "type": "constructor"
        },
        {
          "anonymous": false,
          "inputs": [
            {
              "indexed": false,
              "internalType": "address",
              "name": "member",
              "type": "address"
            }
          ],
          "name": "MemberAdded",
          "type": "event"
        },
        {
          "anonymous": false,
          "inputs": [
            {
              "indexed": false,
              "internalType": "address",
              "name": "member",
              "type": "address"
            }
          ],
          "name": "MemberRemoved",
          "type": "event"
        },
        {
          "anonymous": false,
          "inputs": [
            {
              "indexed": true,
              "internalType": "address",
              "name": "owner",
              "type": "address"
            },
            {
              "indexed": true,
              "internalType": "uint256",
              "name": "round",
              "type": "uint256"
            },
            {
              "indexed": false,
              "internalType": "string",
              "name": "ipfsHash",
              "type": "string"
            }
          ],
          "name": "ModelAdded",
          "type": "event"
        },
        {
          "anonymous": false,
          "inputs": [
            {
              "indexed": false,
              "internalType": "bytes32",
              "name": "modelHash",
              "type": "bytes32"
            },
            {
              "indexed": false,
              "internalType": "address",
              "name": "signer",
              "type": "address"
            }
          ],
          "name": "ModelUpdateSubmitted",
          "type": "event"
        },
        {
          "anonymous": false,
          "inputs": [
            {
              "indexed": false,
              "internalType": "bytes32",
              "name": "modelHash",
              "type": "bytes32"
            },
            {
              "indexed": false,
              "internalType": "address",
              "name": "signer",
              "type": "address"
            }
          ],
          "name": "ModelUpdateVerified",
          "type": "event"
        },
        {
          "anonymous": false,
          "inputs": [
            {
              "indexed": false,
              "internalType": "uint256",
              "name": "newRound",
              "type": "uint256"
            }
          ],
          "name": "NewRoundStarted",
          "type": "event"
        },
        {
          "anonymous": false,
          "inputs": [
            {
              "indexed": false,
              "internalType": "uint256",
              "name": "proposalId",
              "type": "uint256"
            },
            {
              "indexed": false,
              "internalType": "address",
              "name": "proposer",
              "type": "address"
            },
            {
              "indexed": false,
              "internalType": "address",
              "name": "member",
              "type": "address"
            },
            {
              "indexed": false,
              "internalType": "bool",
              "name": "add",
              "type": "bool"
            }
          ],
          "name": "ProposalCreated",
          "type": "event"
        },
        {
          "anonymous": false,
          "inputs": [
            {
              "indexed": false,
              "internalType": "uint256",
              "name": "proposalId",
              "type": "uint256"
            },
            {
              "indexed": false,
              "internalType": "address",
              "name": "signer",
              "type": "address"
            }
          ],
          "name": "ProposalSigned",
          "type": "event"
        },
        {
          "inputs": [
            {
              "internalType": "string",
              "name": "_ipfsHash",
              "type": "string"
            }
          ],
          "name": "addModel",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "address",
              "name": "",
              "type": "address"
            },
            {
              "internalType": "uint256",
              "name": "",
              "type": "uint256"
            }
          ],
          "name": "clientHistory",
          "outputs": [
            {
              "internalType": "address",
              "name": "owner",
              "type": "address"
            },
            {
              "internalType": "uint256",
              "name": "round",
              "type": "uint256"
            },
            {
              "internalType": "uint256",
              "name": "timestamp",
              "type": "uint256"
            },
            {
              "internalType": "string",
              "name": "ipfsHash",
              "type": "string"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "string",
              "name": "",
              "type": "string"
            }
          ],
          "name": "existIPFS",
          "outputs": [
            {
              "internalType": "bool",
              "name": "",
              "type": "bool"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "address",
              "name": "_address",
              "type": "address"
            }
          ],
          "name": "getMemberStatus",
          "outputs": [
            {
              "internalType": "bool",
              "name": "member",
              "type": "bool"
            },
            {
              "internalType": "bool",
              "name": "whitelisted",
              "type": "bool"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [],
          "name": "getMembers",
          "outputs": [
            {
              "internalType": "address[]",
              "name": "",
              "type": "address[]"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [],
          "name": "getModelCount",
          "outputs": [
            {
              "internalType": "uint256",
              "name": "",
              "type": "uint256"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "address",
              "name": "member",
              "type": "address"
            },
            {
              "internalType": "uint256",
              "name": "_round",
              "type": "uint256"
            }
          ],
          "name": "getModelIPFSHash",
          "outputs": [
            {
              "internalType": "string",
              "name": "",
              "type": "string"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "address",
              "name": "",
              "type": "address"
            }
          ],
          "name": "isMember",
          "outputs": [
            {
              "internalType": "bool",
              "name": "",
              "type": "bool"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "address",
              "name": "_address",
              "type": "address"
            }
          ],
          "name": "isWhitelisted",
          "outputs": [
            {
              "internalType": "bool",
              "name": "",
              "type": "bool"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "uint256",
              "name": "",
              "type": "uint256"
            }
          ],
          "name": "members",
          "outputs": [
            {
              "internalType": "address",
              "name": "",
              "type": "address"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [],
          "name": "modelCount",
          "outputs": [
            {
              "internalType": "uint256",
              "name": "",
              "type": "uint256"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "bytes32",
              "name": "",
              "type": "bytes32"
            }
          ],
          "name": "modelUpdates",
          "outputs": [
            {
              "internalType": "bytes32",
              "name": "modelHash",
              "type": "bytes32"
            },
            {
              "internalType": "address",
              "name": "signer",
              "type": "address"
            },
            {
              "internalType": "bool",
              "name": "verified",
              "type": "bool"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [],
          "name": "proposalCount",
          "outputs": [
            {
              "internalType": "uint256",
              "name": "",
              "type": "uint256"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "uint256",
              "name": "",
              "type": "uint256"
            }
          ],
          "name": "proposals",
          "outputs": [
            {
              "internalType": "address",
              "name": "proposer",
              "type": "address"
            },
            {
              "internalType": "address",
              "name": "member",
              "type": "address"
            },
            {
              "internalType": "bool",
              "name": "add",
              "type": "bool"
            },
            {
              "internalType": "uint256",
              "name": "signatures",
              "type": "uint256"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "address",
              "name": "member",
              "type": "address"
            }
          ],
          "name": "proposeAddMember",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "address",
              "name": "member",
              "type": "address"
            }
          ],
          "name": "proposeRemoveMember",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [],
          "name": "round",
          "outputs": [
            {
              "internalType": "uint256",
              "name": "",
              "type": "uint256"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "uint256",
              "name": "proposalId",
              "type": "uint256"
            }
          ],
          "name": "signProposal",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "uint256",
              "name": "",
              "type": "uint256"
            },
            {
              "internalType": "address",
              "name": "",
              "type": "address"
            }
          ],
          "name": "signatures",
          "outputs": [
            {
              "internalType": "bool",
              "name": "",
              "type": "bool"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [],
          "name": "startNewRound",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "bytes",
              "name": "newModelData",
              "type": "bytes"
            },
            {
              "internalType": "bytes",
              "name": "signature",
              "type": "bytes"
            }
          ],
          "name": "submitModelUpdate",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "bytes32",
              "name": "",
              "type": "bytes32"
            }
          ],
          "name": "submittedModels",
          "outputs": [
            {
              "internalType": "bool",
              "name": "",
              "type": "bool"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [],
          "name": "threshold",
          "outputs": [
            {
              "internalType": "uint256",
              "name": "",
              "type": "uint256"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "bytes32",
              "name": "modelHash",
              "type": "bytes32"
            }
          ],
          "name": "verifyModelUpdate",
          "outputs": [],
          "stateMutability": "nonpayable",
          "type": "function"
        },
        {
          "inputs": [
            {
              "internalType": "address",
              "name": "",
              "type": "address"
            }
          ],
          "name": "whitelist",
          "outputs": [
            {
              "internalType": "bool",
              "name": "",
              "type": "bool"
            }
          ],
          "stateMutability": "view",
          "type": "function"
        }
      ]
    """)
MemberMgt = web3.eth.contract(address=contract_address, abi=MemberMgtAbi)
# print(10**18)

# Select an account to use for transactions (you can change the index as needed)
account_index = 0
private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
# account = web3.eth.account.privateKeyToAccount(private_key)
account = web3.eth.account.from_key(private_key)
web3.eth.defaultAccount = account.address

# IPFS client setup
# ipfs_client = ipfshttpclient.connect()


def appendFile(fileName, msg):
    with open(fileName, "a") as f:
        f.write(msg + "\n")


def add_to_ipfs(provider, model):
    ipfs_client = ipfshttpclient.Client("/dns/ipfs.infura.io/tcp/5001/https")
    with open(provider, "wb") as model_file:
        obj_out = {"model_provider": provider, "model": model}
        pickle.dump(obj_out, model_file)
    ipfsFile = ipfs_client.add(provider)
    return ipfsFile["Hash"]


def get_from_ipfs(account):
    ipfs_client = ipfshttpclient.Client("/dns/ipfs.infura.io/tcp/5001/https")
    rounds = MemberMgt.functions.round().call()
    model_info = MemberMgt.functions.clientHistory(account, rounds).call()
    _ = ipfs_client.get(model_info[3])
    return model_info[3]


def estimate_gas_and_time(function, *args):
    start_time = time.time()
    gas_estimate = function.estimateGas({"from": account.address}, *args)
    tx_hash = function.transact({"from": account.address}, *args)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    end_time = time.time()
    execution_time = end_time - start_time
    return gas_estimate, execution_time, receipt


def log_to_csv(data, filename="blockchain_logs.csv"):
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


def start_new_round():
    estimate = MemberMgt.functions.startNewRound().estimateGas(
        {"from": web3.eth.defaultAccount}
    )
    print("startNewRound Function Gas cost", estimate)
    start_time = time.time()
    tx_hash = MemberMgt.functions.startNewRound().transact(
        {"from": web3.eth.defaultAccount}
    )
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    print("--- %s seconds for start new round ---" % (time.time() - start_time))
    return receipt


def propose_add_member(member_address):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(
        MemberMgt.functions.proposeAddMember, member_address
    )
    log_to_csv(["propose_add_member", gas_estimate, execution_time])
    return receipt


def propose_remove_member(member_address):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(
        MemberMgt.functions.proposeRemoveMember, member_address
    )
    log_to_csv(["propose_remove_member", gas_estimate, execution_time])
    return receipt


def sign_proposal(proposal_id):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(
        MemberMgt.functions.signProposal, proposal_id
    )
    log_to_csv(["sign_proposal", gas_estimate, execution_time])
    return receipt


def is_whitelisted(address):
    return MemberMgt.functions.isWhitelisted(address).call()


def submit_model_update(model_hash, signature):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(
        MemberMgt.functions.submitModelUpdate, model_hash, signature
    )
    log_to_csv(["submit_model_update", gas_estimate, execution_time])
    return receipt


def verify_model_update(model_hash):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(
        MemberMgt.functions.verifyModelUpdate, model_hash
    )
    log_to_csv(["verify_model_update", gas_estimate, execution_time])
    return receipt


def verify_signature(model_hash, signature, signer):
    message_hash = Web3.solidityKeccak(["bytes32"], [model_hash])
    message_hash_bytes = bytes.fromhex(
        message_hash.hex()[2:]
    )  # Convert hex string to bytes (excluding "0x" prefix)
    recovered_address = web3.eth.account.recover_message(
        message_hash_bytes, signature=signature
    )
    return recovered_address == signer


def aggregate_updates(updates):
    valid_updates = [
        update
        for update in updates
        if verify_signature(update["model_hash"], update["signature"], update["signer"])
    ]
    aggregated_update = sum([update["model_update"] for update in valid_updates]) / len(
        valid_updates
    )
    return aggregated_update


def weighted_average(metrics):
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    examples = sum([num_examples for num_examples, _ in metrics])
    if examples > 0:
        return {"accuracy": sum(accuracies) / examples}
    else:
        return {"accuracy": 0.0}  # Handle the case where no examples were processed


""" def fit_metrics_aggregation_fn(metrics):
    accuracies = [num_examples * m.get("accuracy", 0.0) for num_examples, m in metrics]
    examples = sum([num_examples for num_examples, _ in metrics])
    if examples > 0:
        return {"accuracy": sum(accuracies) / examples}
    else:
        return {"accuracy": 0.0}  # Handle the case where no examples were processed """

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,  # vv
    ),
)

if __name__ == "__main__":
    # Initialize CSV file with headers
    with open("Serverblockchain_logs.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ServerFunction", "SGasEstimate", "SExecutionTime"])
