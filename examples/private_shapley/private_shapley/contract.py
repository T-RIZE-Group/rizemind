"""Interface for the PrivateShapley smart contract."""

import json
import os
from typing import List, Tuple, Dict, Optional
from dotenv import load_dotenv
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError

# Load environment variables
load_dotenv()
CONTRACT_ADDRESS = os.getenv("PRIVATE_SHAPLEY_CONTRACT")

# Contract ABI - This is a simplified ABI, you might need to expand it based on your contract
CONTRACT_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "inputs": [
            {
                "internalType": "address",
                "name": "trainer",
                "type": "address"
            }
        ],
        "name": "registerTrainer",
        "outputs": [
            {
                "internalType": "uint8",
                "name": "",
                "type": "uint8"
            }
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "bytes32",
                "name": "coalitionId",
                "type": "bytes32"
            },
            {
                "internalType": "bytes32",
                "name": "bitfield",
                "type": "bytes32"
            },
            {
                "internalType": "bytes32",
                "name": "merkleRoot",
                "type": "bytes32"
            }
        ],
        "name": "publishCoalitionData",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "bytes32",
                "name": "coalitionId",
                "type": "bytes32"
            },
            {
                "internalType": "uint256",
                "name": "result",
                "type": "uint256"
            }
        ],
        "name": "publishResult",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "uint256",
                "name": "roundId",
                "type": "uint256"
            },
            {
                "internalType": "bytes32",
                "name": "coalitionId",
                "type": "bytes32"
            },
            {
                "internalType": "bytes32",
                "name": "nonce",
                "type": "bytes32"
            },
            {
                "internalType": "bytes32[]",
                "name": "merkleProof",
                "type": "bytes32[]"
            }
        ],
        "name": "claimReward",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {
                "internalType": "address",
                "name": "trainer",
                "type": "address"
            }
        ],
        "name": "isRegisteredTrainer",
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
                "internalType": "bytes32",
                "name": "coalitionId",
                "type": "bytes32"
            },
            {
                "internalType": "address",
                "name": "trainer",
                "type": "address"
            }
        ],
        "name": "isTrainerInCoalition",
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
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "bytes32",
                "name": "coalitionId",
                "type": "bytes32"
            },
            {
                "indexed": False,
                "internalType": "bytes32",
                "name": "bitfield",
                "type": "bytes32"
            },
            {
                "indexed": False,
                "internalType": "bytes32",
                "name": "merkleRoot",
                "type": "bytes32"
            }
        ],
        "name": "CoalitionDataPublished",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "bytes32",
                "name": "coalitionId",
                "type": "bytes32"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "result",
                "type": "uint256"
            }
        ],
        "name": "ResultPublished",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "trainer",
                "type": "address"
            },
            {
                "indexed": True,
                "internalType": "bytes32",
                "name": "coalitionId",
                "type": "bytes32"
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "reward",
                "type": "uint256"
            }
        ],
        "name": "RewardClaimed",
        "type": "event"
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "address",
                "name": "trainer",
                "type": "address"
            },
            {
                "indexed": False,
                "internalType": "uint8",
                "name": "index",
                "type": "uint8"
            }
        ],
        "name": "TrainerRegistered",
        "type": "event"
    }
]

class PrivateShapleyContract:
    """Interface to the PrivateShapley smart contract."""
    
    def __init__(self, web3_provider: str = "http://127.0.0.1:8545"):
        """Initialize the contract interface.
        
        Args:
            web3_provider: URL of the Web3 provider
        """
        if not CONTRACT_ADDRESS:
            raise ValueError("PrivateShapley contract address not found in environment variables")
        
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(CONTRACT_ADDRESS),
            abi=CONTRACT_ABI
        )
        self.owner_address = None
    
    def set_owner(self, address: str):
        """Set the contract owner address for transactions.
        
        Args:
            address: The owner's Ethereum address
        """
        self.owner_address = Web3.to_checksum_address(address)
    
    # def register_trainer(self, trainer_address: str, private_key: str) -> int:
    #     """Register a new trainer in the contract.
        
    #     Args:
    #         trainer_address: The trainer's Ethereum address
    #         private_key: Private key for the owner account
            
    #     Returns:
    #         The assigned trainer index
    #     """
    #     if not self.owner_address:
    #         raise ValueError("Owner address not set. Call set_owner first.")
        
    #     trainer_address = Web3.to_checksum_address(trainer_address)
        
    #     # Check if trainer is already registered
    #     try:
    #         if self.contract.functions.isRegisteredTrainer(trainer_address).call():
    #             print(f"Trainer {trainer_address} already registered")
    #             # Get the trainer's index
    #             for i in range(1, 256):  # Max trainers is 255
    #                 try:
    #                     stored_address = self.contract.functions.indexToAddress(i).call()
    #                     if stored_address.lower() == trainer_address.lower():
    #                         return i
    #                 except (ContractLogicError, ValueError):
    #                     continue
    #             raise ValueError(f"Could not find index for registered trainer {trainer_address}")
    #     except (ContractLogicError, ValueError):
    #         pass
        
    #     # Build and send transaction
    #     tx = self.contract.functions.registerTrainer(trainer_address).build_transaction({
    #         'from': self.owner_address,
    #         'nonce': self.w3.eth.get_transaction_count(self.owner_address),
    #         'gas': 200000,
    #         'gasPrice': self.w3.eth.gas_price
    #     })
        
    #     signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
    #     tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    #     tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
    #     # Get the trainer's index from the event logs
    #     for log in tx_receipt['logs']:
    #         try:
    #             event = self.contract.events.TrainerRegistered().process_log(log)
    #             if event and event['args']['trainer'].lower() == trainer_address.lower():
    #                 return event['args']['index']
    #         except:
    #             continue
        
    #     raise ValueError("Failed to register trainer")
    
    
    # def register_trainer(self, trainer_address: str, private_key: str) -> int:
    #     """Register a new trainer in the contract.
        
    #     Args:
    #         trainer_address: The trainer's Ethereum address
    #         private_key: Private key for the owner account
            
    #     Returns:
    #         The assigned trainer index
    #     """
    #     if not self.owner_address:
    #         raise ValueError("Owner address not set. Call set_owner first.")
        
    #     print(f"Attempting to register trainer at address: {trainer_address}")
        
    #     try:
    #         trainer_address = Web3.to_checksum_address(trainer_address)
    #     except ValueError as e:
    #         print(f"Invalid Ethereum address format: {e}")
    #         raise
        
    #     # Check if trainer is already registered
    #     try:
    #         print("Checking if trainer is already registered...")
    #         if self.contract.functions.isRegisteredTrainer(trainer_address).call():
    #             print(f"Trainer {trainer_address} already registered")
    #             # Get the trainer's index
    #             print("Searching for trainer's index...")
    #             for i in range(1, 256):  # Max trainers is 255
    #                 try:
    #                     stored_address = self.contract.functions.indexToAddress(i).call()
    #                     if stored_address.lower() == trainer_address.lower():
    #                         print(f"Found trainer index: {i}")
    #                         return i
    #                 except (ContractLogicError, ValueError) as e:
    #                     print(f"Error checking index {i}: {e}")
    #                     continue
    #             print("Could not find trainer's index in registered addresses")
    #             raise ValueError(f"Could not find index for registered trainer {trainer_address}")
    #     except (ContractLogicError, ValueError) as e:
    #         print(f"Error checking trainer registration: {e}")
    #         pass
        
    #     print("Building transaction...")
    #     try:
    #         # Build and send transaction
    #         tx = self.contract.functions.registerTrainer(trainer_address).build_transaction({
    #             'from': self.owner_address,
    #             'nonce': self.w3.eth.get_transaction_count(self.owner_address),
    #             'gas': 200000,
    #             'gasPrice': self.w3.eth.gas_price
    #         })
            
    #         print("Signing transaction...")
    #         # Sign the transaction
    #         signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
            
    #         print("Sending transaction...")
    #         # Use raw_transaction instead of rawTransaction
    #         if hasattr(signed_tx, 'raw_transaction'):
    #             tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    #         else:
    #             # Fallback for older versions of Web3.py
    #             tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
    #         print(f"Transaction sent with hash: {tx_hash.hex()}")
    #         print("Waiting for transaction receipt...")
    #         tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
    #         print(f"Transaction mined in block: {tx_receipt['blockNumber']}")
            
    #         # Get the trainer's index from the event logs
    #         print("Searching event logs for trainer index...")
    #         for log in tx_receipt['logs']:
    #             try:
    #                 event = self.contract.events.TrainerRegistered().process_log(log)
    #                 if event and event['args']['trainer'].lower() == trainer_address.lower():
    #                     trainer_index = event['args']['index']
    #                     print(f"Found trainer index in logs: {trainer_index}")
    #                     return trainer_index
    #             except Exception as e:
    #                 print(f"Error processing log: {e}")
    #                 continue
            
    #         print("Could not find TrainerRegistered event in logs")
            
    #     except Exception as e:
    #         print(f"Error during transaction execution: {e}")
    #         raise
        
    #     raise ValueError("Failed to register trainer - no registration event found")
    
    
    def register_trainer(self, trainer_address: str, private_key: str) -> int:
        """Register a new trainer in the contract."""
        if not self.owner_address:
            raise ValueError("Owner address not set. Call set_owner first.")
        
        print(f"Attempting to register trainer at address: {trainer_address}")
        
        try:
            trainer_address = Web3.to_checksum_address(trainer_address)
        except ValueError as e:
            print(f"Invalid Ethereum address format: {e}")
            raise
        
        # Check if trainer is already registered
        try:
            if self.contract.functions.isRegisteredTrainer(trainer_address).call():
                print(f"Trainer {trainer_address} already registered")
                # Get the trainer's index
                for i in range(1, 256):  # Max trainers is 255
                    try:
                        stored_address = self.contract.functions.indexToAddress(i).call()
                        if stored_address.lower() == trainer_address.lower():
                            return i
                    except (ContractLogicError, ValueError):
                        continue
                raise ValueError(f"Could not find index for registered trainer {trainer_address}")
        except (ContractLogicError, ValueError) as e:
            print(f"Error checking trainer registration: {e}")
        
        try:
            # Build and send transaction
            tx = self.contract.functions.registerTrainer(trainer_address).build_transaction({
                'from': self.owner_address,
                'nonce': self.w3.eth.get_transaction_count(self.owner_address),
                'gas': 300000,  # Increased gas limit
                'gasPrice': self.w3.eth.gas_price
            })
            
            # Sign and send transaction
            signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
            
            if hasattr(signed_tx, 'raw_transaction'):
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            else:
                tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Check if transaction was successful
            if tx_receipt['status'] == 0:
                try:
                    self.contract.functions.registerTrainer(trainer_address).call({
                        'from': self.owner_address
                    })
                except Exception as call_ex:
                    print(f"Transaction reverted: {str(call_ex)}")
                raise ValueError("Transaction reverted")
            
            # Get the trainer's index from the event logs
            for log in tx_receipt['logs']:
                try:
                    event = self.contract.events.TrainerRegistered().process_log(log)
                    if event and event['args']['trainer'].lower() == trainer_address.lower():
                        return event['args']['index']
                except Exception:
                    continue
            
            # Fallback - check if registered and get index manually
            if self.contract.functions.isRegisteredTrainer(trainer_address).call():
                for i in range(1, 256):
                    try:
                        stored_address = self.contract.functions.indexToAddress(i).call()
                        if stored_address.lower() == trainer_address.lower():
                            return i
                    except Exception:
                        continue
                        
                # Last resort - use trainer count
                trainer_count = self.contract.functions.trainerCount().call()
                if self.contract.functions.isRegisteredTrainer(trainer_address).call():
                    return trainer_count
            
        except Exception as e:
            print(f"Error during transaction execution: {e}")
            raise
        
        raise ValueError("Failed to register trainer - no registration event found")
    # def publish_coalition_data(self, coalition_id: bytes, bitfield: bytes, merkle_root: bytes, private_key: str) -> None:
    #     """Publish coalition data to the contract.
        
    #     Args:
    #         coalition_id: Unique identifier for the coalition
    #         bitfield: Bitfield representing trainer membership
    #         merkle_root: Merkle root of commitments
    #         private_key: Private key for the owner account
    #     """
    #     if not self.owner_address:
    #         raise ValueError("Owner address not set. Call set_owner first.")
        
    #     # Build and send transaction
    #     tx = self.contract.functions.publishCoalitionData(
    #         coalition_id,
    #         bitfield,
    #         merkle_root
    #     ).build_transaction({
    #         'from': self.owner_address,
    #         'nonce': self.w3.eth.get_transaction_count(self.owner_address),
    #         'gas': 200000,
    #         'gasPrice': self.w3.eth.gas_price
    #     })
        
    #     signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
    #     tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    #     self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    
    def publish_coalition_data(self, coalition_id: bytes, bitfield: bytes, merkle_root: bytes, private_key: str) -> None:
        """Publish coalition data to the contract.
        
        Args:
            coalition_id: Unique identifier for the coalition
            bitfield: Bitfield representing trainer membership
            merkle_root: Merkle root of commitments
            private_key: Private key for the owner account
        """
        if not self.owner_address:
            raise ValueError("Owner address not set. Call set_owner first.")
        
        # Build and send transaction
        tx = self.contract.functions.publishCoalitionData(
            coalition_id,
            bitfield,
            merkle_root
        ).build_transaction({
            'from': self.owner_address,
            'nonce': self.w3.eth.get_transaction_count(self.owner_address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
        
        # Use raw_transaction instead of rawTransaction
        if hasattr(signed_tx, 'raw_transaction'):
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        else:
            # Fallback for older versions of Web3.py
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    # def publish_result(self, coalition_id: bytes, result: int, address: str, private_key: str) -> None:
    #     """Publish a result for a coalition.
        
    #     Args:
    #         coalition_id: Unique identifier for the coalition
    #         result: The evaluation score (multiplied by 10^6 for precision)
    #         address: The publisher's address
    #         private_key: Private key for the publisher account
    #     """
    #     address = Web3.to_checksum_address(address)
        
    #     # Build and send transaction
    #     tx = self.contract.functions.publishResult(
    #         coalition_id,
    #         result
    #     ).build_transaction({
    #         'from': address,
    #         'nonce': self.w3.eth.get_transaction_count(address),
    #         'gas': 200000,
    #         'gasPrice': self.w3.eth.gas_price
    #     })
        
    #     signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
    #     tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    #     self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    
    def publish_result(self, coalition_id: bytes, result: int, address: str, private_key: str) -> None:
        """Publish a result for a coalition.
        
        Args:
            coalition_id: Unique identifier for the coalition
            result: The evaluation score (multiplied by 10^6 for precision)
            address: The publisher's address
            private_key: Private key for the publisher account
        """
        address = Web3.to_checksum_address(address)
        
        # Build and send transaction
        tx = self.contract.functions.publishResult(
            coalition_id,
            result
        ).build_transaction({
            'from': address,
            'nonce': self.w3.eth.get_transaction_count(address),
            'gas': 200000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
        
        # Use raw_transaction instead of rawTransaction
        if hasattr(signed_tx, 'raw_transaction'):
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        else:
            # Fallback for older versions of Web3.py
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    
    # def claim_reward(self, round_id: int, coalition_id: bytes, nonce: bytes, merkle_proof: List[bytes], 
    #                  address: str, private_key: str) -> None:
    #     """Claim a reward for participating in a coalition.
        
    #     Args:
    #         round_id: The training round ID
    #         coalition_id: Unique identifier for the coalition
    #         nonce: The secret nonce used for commitment
    #         merkle_proof: Merkle proof of participation
    #         address: The trainer's address
    #         private_key: Private key for the trainer account
    #     """
    #     address = Web3.to_checksum_address(address)
        
    #     # Build and send transaction
    #     tx = self.contract.functions.claimReward(
    #         round_id,
    #         coalition_id,
    #         nonce,
    #         merkle_proof
    #     ).build_transaction({
    #         'from': address,
    #         'nonce': self.w3.eth.get_transaction_count(address),
    #         'gas': 300000,
    #         'gasPrice': self.w3.eth.gas_price
    #     })
        
    #     signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
    #     tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
    #     self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def claim_reward(self, round_id: int, coalition_id: bytes, nonce: bytes, merkle_proof: List[bytes], 
                 address: str, private_key: str) -> None:
        """Claim a reward for participating in a coalition.
        
        Args:
            round_id: The training round ID
            coalition_id: Unique identifier for the coalition
            nonce: The secret nonce used for commitment
            merkle_proof: Merkle proof of participation
            address: The trainer's address
            private_key: Private key for the trainer account
        """
        address = Web3.to_checksum_address(address)
        
        # Build and send transaction
        tx = self.contract.functions.claimReward(
            round_id,
            coalition_id,
            nonce,
            merkle_proof
        ).build_transaction({
            'from': address,
            'nonce': self.w3.eth.get_transaction_count(address),
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price
        })
        
        signed_tx = self.w3.eth.account.sign_transaction(tx, private_key)
        
        # Use raw_transaction instead of rawTransaction
        if hasattr(signed_tx, 'raw_transaction'):
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        else:
            # Fallback for older versions of Web3.py
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
    
    def is_registered_trainer(self, address: str) -> bool:
        """Check if an address is a registered trainer.
        
        Args:
            address: The address to check
            
        Returns:
            True if the address is a registered trainer, False otherwise
        """
        address = Web3.to_checksum_address(address)
        return self.contract.functions.isRegisteredTrainer(address).call()
    
    def is_trainer_in_coalition(self, coalition_id: bytes, address: str) -> bool:
        """Check if a trainer is part of a coalition.
        
        Args:
            coalition_id: The coalition identifier
            address: The trainer's address
            
        Returns:
            True if the trainer is in the coalition, False otherwise
        """
        address = Web3.to_checksum_address(address)
        return self.contract.functions.isTrainerInCoalition(coalition_id, address).call()
    
    def get_coalition_result_events(self, from_block: int = 0) -> List[Dict]:
        """Get all ResultPublished events.
        
        Args:
            from_block: Block number to start searching from
            
        Returns:
            List of event data dictionaries
        """
        result_filter = self.contract.events.ResultPublished.create_filter(
            fromBlock=from_block
        )
        events = result_filter.get_all_entries()
        
        return [
            {
                'coalition_id': event['args']['coalitionId'],
                'result': event['args']['result'],
                'block_number': event['blockNumber']
            }
            for event in events
        ]