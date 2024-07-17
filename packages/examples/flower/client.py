import flwr as fl
import tensorflow as tf
from web3 import Web3
from eth_account import Account
import json
import hashlib
import ipfshttpclient
import pickle
import time
import csv

# Connect to Ethereum node
import requests
#w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))

# Smart contract address and ABI
contract_address = '0x5FbDB2315678afecb367f032d93F642f64180aa3'
web3 = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:7545"))
MemberMgtAbi = json.loads(""" [
        
                {
               "inputs":[
                  {
                     "internalType":"address[]",
                     "name":"initialMembers",
                     "type":"address[]"
                  },
                  {
                     "internalType":"uint256",
                     "name":"initialThreshold",
                     "type":"uint256"
                  }
               ],
               "stateMutability":"nonpayable",
               "type":"constructor"
            },
            {
               "anonymous":false,
               "inputs":[
                  {
                     "indexed":false,
                     "internalType":"address",
                     "name":"member",
                     "type":"address"
                  }
               ],
               "name":"MemberAdded",
               "type":"event"
            },
            {
               "anonymous":false,
               "inputs":[
                  {
                     "indexed":false,
                     "internalType":"address",
                     "name":"member",
                     "type":"address"
                  }
               ],
               "name":"MemberRemoved",
               "type":"event"
            },
            {
               "anonymous":false,
               "inputs":[
                  {
                     "indexed":false,
                     "internalType":"uint256",
                     "name":"proposalId",
                     "type":"uint256"
                  },
                  {
                     "indexed":false,
                     "internalType":"address",
                     "name":"proposer",
                     "type":"address"
                  },
                  {
                     "indexed":false,
                     "internalType":"address",
                     "name":"member",
                     "type":"address"
                  },
                  {
                     "indexed":false,
                     "internalType":"bool",
                     "name":"add",
                     "type":"bool"
                  }
               ],
               "name":"ProposalCreated",
               "type":"event"
            },
            {
               "anonymous":false,
               "inputs":[
                  {
                     "indexed":false,
                     "internalType":"uint256",
                     "name":"proposalId",
                     "type":"uint256"
                  },
                  {
                     "indexed":false,
                     "internalType":"address",
                     "name":"signer",
                     "type":"address"
                  }
               ],
               "name":"ProposalSigned",
               "type":"event"
            },
            {
               "inputs":[
                  {
                     "internalType":"address",
                     "name":"_address",
                     "type":"address"
                  }
               ],
               "name":"getMemberStatus",
               "outputs":[
                  {
                     "internalType":"bool",
                     "name":"member",
                     "type":"bool"
                  },
                  {
                     "internalType":"bool",
                     "name":"whitelisted",
                     "type":"bool"
                  }
               ],
               "stateMutability":"view",
               "type":"function"
            },
            {
               "inputs":[
                  
               ],
               "name":"getMembers",
               "outputs":[
                  {
                     "internalType":"address[]",
                     "name":"",
                     "type":"address[]"
                  }
               ],
               "stateMutability":"view",
               "type":"function"
            },
            {
               "inputs":[
                  {
                     "internalType":"address",
                     "name":"",
                     "type":"address"
                  }
               ],
               "name":"isMember",
               "outputs":[
                  {
                     "internalType":"bool",
                     "name":"",
                     "type":"bool"
                  }
               ],
               "stateMutability":"view",
               "type":"function"
            },
            {
               "inputs":[
                  {
                     "internalType":"address",
                     "name":"_address",
                     "type":"address"
                  }
               ],
               "name":"isWhitelisted",
               "outputs":[
                  {
                     "internalType":"bool",
                     "name":"",
                     "type":"bool"
                  }
               ],
               "stateMutability":"view",
               "type":"function"
            },
            {
               "inputs":[
                  {
                     "internalType":"uint256",
                     "name":"",
                     "type":"uint256"
                  }
               ],
               "name":"members",
               "outputs":[
                  {
                     "internalType":"address",
                     "name":"",
                     "type":"address"
                  }
               ],
               "stateMutability":"view",
               "type":"function"
            },
            {
               "inputs":[
                  
               ],
               "name":"proposalCount",
               "outputs":[
                  {
                     "internalType":"uint256",
                     "name":"",
                     "type":"uint256"
                  }
               ],
               "stateMutability":"view",
               "type":"function"
            },
            {
               "inputs":[
                  {
                     "internalType":"uint256",
                     "name":"",
                     "type":"uint256"
                  }
               ],
               "name":"proposals",
               "outputs":[
                  {
                     "internalType":"address",
                     "name":"proposer",
                     "type":"address"
                  },
                  {
                     "internalType":"address",
                     "name":"member",
                     "type":"address"
                  },
                  {
                     "internalType":"bool",
                     "name":"add",
                     "type":"bool"
                  },
                  {
                     "internalType":"uint256",
                     "name":"signatures",
                     "type":"uint256"
                  }
               ],
               "stateMutability":"view",
               "type":"function"
            },
            {
               "inputs":[
                  {
                     "internalType":"address",
                     "name":"member",
                     "type":"address"
                  }
               ],
               "name":"proposeAddMember",
               "outputs":[
                  
               ],
               "stateMutability":"nonpayable",
               "type":"function"
            },
            {
               "inputs":[
                  {
                     "internalType":"address",
                     "name":"member",
                     "type":"address"
                  }
               ],
               "name":"proposeRemoveMember",
               "outputs":[
                  
               ],
               "stateMutability":"nonpayable",
               "type":"function"
            },
            {
               "inputs":[
                  {
                     "internalType":"uint256",
                     "name":"proposalId",
                     "type":"uint256"
                  }
               ],
               "name":"signProposal",
               "outputs":[
                  
               ],
               "stateMutability":"nonpayable",
               "type":"function"
            },
            {
               "inputs":[
                  {
                     "internalType":"uint256",
                     "name":"",
                     "type":"uint256"
                  },
                  {
                     "internalType":"address",
                     "name":"",
                     "type":"address"
                  }
               ],
               "name":"signatures",
               "outputs":[
                  {
                     "internalType":"bool",
                     "name":"",
                     "type":"bool"
                  }
               ],
               "stateMutability":"view",
               "type":"function"
            },
            {
               "inputs":[
                  
               ],
               "name":"threshold",
               "outputs":[
                  {
                     "internalType":"uint256",
                     "name":"",
                     "type":"uint256"
                  }
               ],
               "stateMutability":"view",
               "type":"function"
            },
            {
               "inputs":[
                  {
                     "internalType":"address",
                     "name":"",
                     "type":"address"
                  }
               ],
               "name":"whitelist",
               "outputs":[
                  {
                     "internalType":"bool",
                     "name":"",
                     "type":"bool"
                  }
               ],
               "stateMutability":"view",
               "type":"function"
            }
      ]
    """)
MemberMgt = web3.eth.contract(address=contract_address, abi=MemberMgtAbi)

# Select an account to use for transactions (you can change the index as needed)
account_index = 1
private_key = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
account = web3.eth.account.from_key(private_key)

#account = web3.eth.account.privateKeyToAccount(private_key)
web3.eth.defaultAccount = account.address

def appendFile(fileName, msg):
    f = open(fileName, "a")
    f.write(msg)
    f.write("\n")
    f.close()
# TensorFlow setup
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()



def add_to_ipfs(provider, model):
    ipfs_client = ipfshttpclient.Client('/dns/ipfs.infura.io/tcp/5001/https')
    model_file = open(provider, "wb")
    obj_out = {"model_provider": provider, "model": model}
    pickle.dump(obj_out, model_file)
    model_file.close()
    ipfsFile = ipfs_client.add(provider)
    return ipfsFile['Hash']

def get_from_ipfs(account):
    ipfs_client = ipfshttpclient.Client('/dns/ipfs.infura.io/tcp/5001/https')
    rounds = MemberMgt.functions.round().call()
    model_info = MemberMgt.functions.clientHistory(account, rounds).call()
    file = ipfs_client.get(model_info[3])
    return model_info[3]

def estimate_gas_and_time(function, *args):
    start_time = time.time()
    gas_estimate = function.estimateGas({'from': account.address}, *args)
    tx_hash = function.transact({'from': account.address}, *args)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    end_time = time.time()
    execution_time = end_time - start_time
    return gas_estimate, execution_time, receipt

def log_to_csv(data, filename="blockchain_logs.csv"):
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)

def hash_model_update(parameters):
    model_bytes = json.dumps(parameters, default=lambda x: x.tolist()).encode('utf-8')
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    return model_hash

def sign_model_update(model_hash):
    message = Web3.solidityKeccak(['bytes32'], [model_hash])
    signed_message = web3.eth.account.sign_message(message, private_key=private_key)
    return signed_message.signature

def start_new_round(ipfs_hash):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.startNewRound, ipfs_hash)
    log_to_csv(["start_new_round", gas_estimate, execution_time])
    return receipt

def submit_model_update(ipfs_hash):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.addModel, ipfs_hash)
    log_to_csv(["submit_model_update", gas_estimate, execution_time])
    return receipt

def verify_model_update(model_hash):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.verifyModelUpdate, model_hash)
    log_to_csv(["verify_model_update", gas_estimate, execution_time])
    return receipt

def propose_add_member(member_address):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.proposeAddMember, member_address)
    log_to_csv(["propose_add_member", gas_estimate, execution_time])
    return receipt

def propose_remove_member(member_address):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.proposeRemoveMember, member_address)
    log_to_csv(["propose_remove_member", gas_estimate, execution_time])
    return receipt

def sign_proposal(proposal_id):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.signProposal, proposal_id)
    log_to_csv(["sign_proposal", gas_estimate, execution_time])
    return receipt

def is_whitelisted(address):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.isWhitelisted, address)
    log_to_csv(["is_whitelisted", gas_estimate, execution_time])
    return MemberMgt.functions.isWhitelisted(address).call()

def get_members():
    gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.getMembers)
    log_to_csv(["get_members", gas_estimate, execution_time])
    return MemberMgt.functions.getMembers().call()

def get_member_status(address):
    gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.getMemberStatus, address)
    log_to_csv(["get_member_status", gas_estimate, execution_time])
    return MemberMgt.functions.getMemberStatus(address).call()

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}

    def submit_model_update(self, parameters):
        model_hash = hash_model_update(parameters)
        ipfs_hash = add_to_ipfs("model_weights.pkl", model)
        gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.addModel, ipfs_hash)
        log_to_csv(["submit_model_update", gas_estimate, execution_time])
        return receipt

    def verify_model_update(self, model_hash):
        gas_estimate, execution_time, receipt = estimate_gas_and_time(MemberMgt.functions.verifyModelUpdate, model_hash)
        log_to_csv(["verify_model_update", gas_estimate, execution_time])
        return receipt

if __name__ == "__main__":
    # Initialize CSV file with headers
    with open("blockchain_logs.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Function", "GasEstimate", "ExecutionTime"])
    
    fl.client.start_client(server_address="127.0.0.1:8080", client=CifarClient().to_client())

